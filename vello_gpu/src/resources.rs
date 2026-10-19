// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent renderer resources shared across frames.

#[cfg(feature = "text")]
use crate::text::GlyphAtlasResources;
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;
use hashbrown::HashMap;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::{
    encode::EncodedPaint,
    image_cache::ImageResource,
    paint::{ImageId, ImageSource},
    pixmap::Pixmap,
};

/// Image allocations for one render, without modifying the retained scene.
/// These are transport resources, independent of optional glyph raster caching.
#[derive(Debug, Default)]
pub(crate) struct PixmapImages {
    by_paint: Vec<Option<ImageId>>,
    pub(crate) uploads: Vec<(ImageId, Arc<Pixmap>)>,
}

impl PixmapImages {
    pub(crate) fn prepare(
        paints: &[EncodedPaint],
        cache: &mut ImageCache,
    ) -> Result<Self, crate::RenderError> {
        let mut images = Self::default();
        let mut seen: HashMap<*const Pixmap, ImageId> = HashMap::default();
        for paint in paints {
            let id = if let EncodedPaint::Image(image) = paint
                && let ImageSource::Pixmap(pixmap) = &image.source
            {
                let key = Arc::as_ptr(pixmap);
                if let Some(id) = seen.get(&key) {
                    Some(*id)
                } else {
                    let id = match cache.allocate(
                        pixmap.width(),
                        pixmap.height(),
                        crate::render::common::IMAGE_PADDING,
                    ) {
                        Ok(id) => id,
                        Err(error) => {
                            images.release(cache);
                            return Err(error.into());
                        }
                    };
                    images.uploads.push((id, Arc::clone(pixmap)));
                    seen.insert(key, id);
                    Some(id)
                }
            } else {
                None
            };
            images.by_paint.push(id);
        }
        Ok(images)
    }

    pub(crate) fn image(&self, paint: usize) -> Result<ImageId, crate::RenderError> {
        self.by_paint
            .get(paint)
            .copied()
            .flatten()
            .ok_or(crate::RenderError::UnpreparedPixmap(paint))
    }

    /// Release CPU allocations after upload/draw commands have been encoded.
    /// The backend must clear the returned regions in command order before reuse.
    pub(crate) fn release(self, cache: &mut ImageCache) -> Vec<ImageResource> {
        self.uploads
            .into_iter()
            .filter_map(|(id, _)| cache.deallocate(id))
            .collect()
    }
}

/// Persistent resources required by Vello GPU for rendering.
///
/// A set of resources must only be used with the renderer instance associated with it.
#[derive(Debug)]
pub struct Resources {
    pub(crate) image_cache: ImageCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    pub(crate) fn new(image_atlas_config: AtlasConfig) -> Self {
        Self {
            image_cache: ImageCache::new_with_config(image_atlas_config),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::{encode::EncodeExt, kurbo::Affine, paint::Image, peniko::ImageSampler};

    fn paints(images: &[Arc<Pixmap>]) -> Vec<EncodedPaint> {
        let mut paints = Vec::new();
        for image in images {
            Image {
                image: ImageSource::Pixmap(Arc::clone(image)),
                sampler: ImageSampler::default(),
            }
            .encode_into(&mut paints, Affine::IDENTITY, None)
            .unwrap();
        }
        paints
    }

    fn allocations(cache: &ImageCache) -> u32 {
        cache
            .atlas_manager()
            .atlas_stats()
            .iter()
            .map(|(_, stats)| stats.allocated_count)
            .sum()
    }

    #[test]
    fn pixmap_preparation_deduplicates_and_releases_without_retaining_scenes() {
        let image = Arc::new(Pixmap::new(12, 12));
        let paints = paints(&[Arc::clone(&image), Arc::clone(&image)]);
        let references = Arc::strong_count(&image);
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (16, 16),
            max_atlases: 1,
            ..Default::default()
        });
        for _ in 0..130 {
            let images = PixmapImages::prepare(&paints, &mut cache).unwrap();
            assert_eq!(images.uploads.len(), 1);
            assert_eq!(images.image(0).unwrap(), images.image(1).unwrap());
            assert_eq!(allocations(&cache), 1);
            assert_eq!(images.release(&mut cache).len(), 1);
            assert_eq!(allocations(&cache), 0);
            assert_eq!(Arc::strong_count(&image), references);
        }
        assert_eq!(cache.atlas_count(), 1);
    }

    #[test]
    fn pixmap_allocation_failure_rolls_back_all_prior_allocations() {
        let paints = paints(&[Arc::new(Pixmap::new(12, 12)), Arc::new(Pixmap::new(12, 12))]);
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (16, 16),
            max_atlases: 1,
            ..Default::default()
        });
        assert!(matches!(
            PixmapImages::prepare(&paints, &mut cache),
            Err(crate::RenderError::AtlasError(_))
        ));
        assert_eq!(allocations(&cache), 0);
    }
}

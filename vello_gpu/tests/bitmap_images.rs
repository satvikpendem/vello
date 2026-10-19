// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual GPU readback regressions; these require an available native WGPU adapter.
//!
//! Set `VELLO_BITMAP_FONT` to exercise an external CBDT/CBLC font instead of the
//! checked-in fixture. `VELLO_BITMAP_OUTPUT` optionally saves the readbacks as PNGs.
#![cfg(not(target_arch = "wasm32"))]

use skrifa::{FontRef, MetadataProvider};
use std::sync::Arc;
use vello_common::{
    kurbo::{Affine, Rect},
    multi_atlas::AtlasConfig,
    paint::{Image, ImageSource},
    peniko::{self, ImageSampler},
    pixmap::Pixmap,
};
use vello_gpu::{MemorySettings, RenderSettings, Resources, Scene};

const SIZE: u16 = 512;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("an actual GPU adapter is required for this test");
        eprintln!("GPU readback adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        Self { device, queue }
    }

    fn renderer(&self, atlas: AtlasConfig) -> (vello_gpu::Renderer, Resources) {
        vello_gpu::Renderer::new_with(
            &self.device,
            &vello_gpu::RenderTargetConfig {
                format: wgpu::TextureFormat::Rgba8Unorm,
                width: SIZE,
                height: SIZE,
            },
            RenderSettings {
                memory_settings: MemorySettings {
                    image_atlas_config: atlas,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    fn texture(&self) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bitmap readback target"),
            size: wgpu::Extent3d {
                width: u32::from(SIZE),
                height: u32::from(SIZE),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn encode(
        &self,
        renderer: &mut vello_gpu::Renderer,
        resources: &mut Resources,
        scene: &Scene,
        target: &wgpu::Texture,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), vello_gpu::RenderError> {
        let size = vello_gpu::RenderSize {
            width: SIZE,
            height: SIZE,
        };
        let depth = vello_gpu::Renderer::create_depth_texture_view(&self.device, &size);
        renderer.render(
            scene,
            resources,
            &self.device,
            &self.queue,
            encoder,
            &size,
            &target.create_view(&Default::default()),
            Some(&depth),
            &vello_gpu::TextureBindings::new(),
            vello_gpu::TargetInit::Clear(Default::default()),
        )
    }

    fn read(&self, target: &wgpu::Texture, encoder: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU bitmap readback"),
            size: u64::from(SIZE) * u64::from(SIZE) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(u32::from(SIZE) * 4),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: u32::from(SIZE),
                height: u32::from(SIZE),
                depth_or_array_layers: 1,
            },
        );
        buffer
    }

    fn pixels(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap()
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let pixels = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        buffer.unmap();
        pixels
    }

    fn render(
        &self,
        renderer: &mut vello_gpu::Renderer,
        resources: &mut Resources,
        scene: &Scene,
    ) -> Vec<u8> {
        let target = self.texture();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(renderer, resources, scene, &target, &mut encoder)
            .unwrap();
        let buffer = self.read(&target, &mut encoder);
        self.queue.submit([encoder.finish()]);
        self.pixels(&buffer)
    }
}

fn font() -> (peniko::FontData, u32) {
    let bytes = std::env::var("VELLO_BITMAP_FONT").map_or_else(
        |_| include_bytes!("../../assets/noto_color_emoji/NotoColorEmoji-CBTF-Subset.ttf").to_vec(),
        |path| std::fs::read(path).unwrap(),
    );
    let parsed = FontRef::from_index(&bytes, 0).unwrap();
    let glyph = parsed
        .charmap()
        .map('🦀')
        .or_else(|| parsed.charmap().map('✅'))
        .unwrap()
        .to_u32();
    (
        peniko::FontData::new(peniko::Blob::new(Arc::new(bytes)), 0),
        glyph,
    )
}

fn glyph_scene(
    resources: &mut Resources,
    font: &peniko::FontData,
    id: u32,
    cache: bool,
    size: f32,
    transform: Affine,
) -> Scene {
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_transform(transform);
    scene
        .glyph_run(resources, font)
        .atlas_cache(cache)
        .font_size(size)
        .fill_glyphs(std::iter::once(glifo::Glyph {
            id,
            x: 140.0,
            y: 300.0,
        }))
        .unwrap();
    scene
}

fn cpu_glyph(font: &peniko::FontData, id: u32, size: f32, transform: Affine) -> Vec<u8> {
    let mut context = vello_cpu::RenderContext::new(SIZE, SIZE);
    let mut resources = vello_cpu::Resources::new();
    context.set_transform(transform);
    context
        .glyph_run(&mut resources, font)
        .atlas_cache(false)
        .font_size(size)
        .fill_glyphs(std::iter::once(glifo::Glyph {
            id,
            x: 140.0,
            y: 300.0,
        }))
        .unwrap();
    let mut image = Pixmap::new(SIZE, SIZE);
    context.render(&mut image, &mut resources);
    image.data_as_u8_slice().to_vec()
}

fn bounds(image: &[u8]) -> [usize; 4] {
    let mut bounds = [usize::MAX, usize::MAX, 0, 0];
    for (index, pixel) in image.as_chunks::<4>().0.iter().enumerate() {
        if pixel[3] > 16 {
            let (x, y) = (index % usize::from(SIZE), index / usize::from(SIZE));
            bounds = [
                bounds[0].min(x),
                bounds[1].min(y),
                bounds[2].max(x),
                bounds[3].max(y),
            ];
        }
    }
    assert_ne!(
        bounds[0],
        usize::MAX,
        "reference/render must contain visible pixels"
    );
    bounds
}

fn compare(actual: &[u8], reference: &[u8], label: &str) {
    let (a, b) = (bounds(actual), bounds(reference));
    for (actual, reference) in a.iter().zip(b) {
        assert!(
            actual.abs_diff(reference) <= 2,
            "{label}: bounds {a:?} != {b:?}"
        );
    }
    let mut total = 0_u64;
    let mut count = 0_u64;
    let mut mismatched = 0_u64;
    for (actual, reference) in actual
        .as_chunks::<4>()
        .0
        .iter()
        .zip(reference.as_chunks::<4>().0.iter())
    {
        if actual[3] > 0 || reference[3] > 0 {
            let error: u64 = actual
                .iter()
                .zip(reference)
                .map(|(a, b)| u64::from(a.abs_diff(*b)))
                .sum();
            total += error;
            count += 1;
            mismatched += u64::from(error > 128);
        }
    }
    let mean = total as f64 / (count * 4) as f64;
    let bad = mismatched as f64 / count as f64;
    assert!(
        mean < 1.0 && bad < 0.01,
        "{label}: mean channel error {mean}, bad-pixel fraction {bad}"
    );
    eprintln!("PASS {label}: bounds={a:?}, mean-channel-error={mean:.3}, bad-pixels={bad:.4}");
    if let Ok(dir) = std::env::var("VELLO_BITMAP_OUTPUT") {
        std::fs::create_dir_all(&dir).unwrap();
        let file =
            std::fs::File::create(std::path::Path::new(&dir).join(format!("{label}.png"))).unwrap();
        let mut encoder = png::Encoder::new(file, u32::from(SIZE), u32::from(SIZE));
        encoder.set_color(png::ColorType::Rgba);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(actual)
            .unwrap();
    }
}

#[test]
fn bitmap_glyphs_match_cpu_across_cache_cutoffs_and_cold_warm_transforms() {
    let gpu = Gpu::new();
    let (font, glyph) = font();
    let rotation = Affine::translate((256.0, 256.0))
        * Affine::rotate(0.3)
        * Affine::translate((-256.0, -256.0));
    let transforms = [
        ("identity", Affine::IDENTITY),
        ("rotation", rotation),
        (
            "mirror",
            Affine::translate((512.0, 0.0)) * Affine::scale_non_uniform(-1.0, 1.0),
        ),
        ("skew", Affine::new([1.0, 0.05, 0.15, 1.0, -20.0, -10.0])),
    ];
    for cache in [false, true] {
        for size in [20.0, 128.0, 129.0, 200.0] {
            for (name, transform) in transforms {
                let (mut renderer, mut resources) = gpu.renderer(AtlasConfig::default());
                let reference = cpu_glyph(&font, glyph, size, transform);
                let scene = glyph_scene(&mut resources, &font, glyph, cache, size, transform);
                compare(
                    &gpu.render(&mut renderer, &mut resources, &scene),
                    &reference,
                    &format!("cold-cache-{cache}-size-{size}-{name}"),
                );
                let warm = glyph_scene(&mut resources, &font, glyph, cache, size, Affine::IDENTITY);
                let _ = gpu.render(&mut renderer, &mut resources, &warm);
                let scene = glyph_scene(&mut resources, &font, glyph, cache, size, transform);
                compare(
                    &gpu.render(&mut renderer, &mut resources, &scene),
                    &reference,
                    &format!("warm-cache-{cache}-size-{size}-{name}"),
                );
            }
        }
    }
}

fn colored_image(color: [u8; 4]) -> Arc<Pixmap> {
    let mut image = Pixmap::new(12, 12);
    for pixel in image.data_as_u8_slice_mut().as_chunks_mut::<4>().0 {
        pixel.copy_from_slice(&color);
    }
    Arc::new(image)
}

fn image_scene(image: Arc<Pixmap>) -> Scene {
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_transform(Affine::translate((100.0, 110.0)));
    scene.set_paint(Image {
        image: ImageSource::Pixmap(image),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, 12.0, 12.0));
    scene
}

#[test]
fn retained_scene_reuploads_without_exhausting_one_image_atlas() {
    let gpu = Gpu::new();
    let (mut renderer, mut resources) = gpu.renderer(AtlasConfig {
        atlas_size: (16, 16),
        max_atlases: 1,
        ..Default::default()
    });
    let image = colored_image([220, 35, 10, 255]);
    let scene = image_scene(Arc::clone(&image));
    let original_strong_count = Arc::strong_count(&image);
    let reference = gpu.render(&mut renderer, &mut resources, &scene);
    for frame in 0..130 {
        let actual = gpu.render(&mut renderer, &mut resources, &scene);
        assert_eq!(actual, reference, "retained scene changed at frame {frame}");
    }
    assert_eq!(
        Arc::strong_count(&image),
        original_strong_count,
        "rendering retained the Pixmap after cleanup"
    );
    assert_eq!(
        &reference[(110 * usize::from(SIZE) + 100) * 4..][..4],
        &[220, 35, 10, 255]
    );
}

#[test]
fn allocation_failure_is_typed_and_does_not_leak_partial_allocations() {
    let gpu = Gpu::new();
    let (mut renderer, mut resources) = gpu.renderer(AtlasConfig {
        atlas_size: (16, 16),
        max_atlases: 1,
        ..Default::default()
    });
    let mut scene = image_scene(colored_image([220, 35, 10, 255]));
    scene.set_paint(Image {
        image: ImageSource::Pixmap(colored_image([10, 35, 220, 255])),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(20.0, 0.0, 32.0, 12.0));
    let target = gpu.texture();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let error = gpu
        .encode(&mut renderer, &mut resources, &scene, &target, &mut encoder)
        .unwrap_err();
    assert!(matches!(error, vello_gpu::RenderError::AtlasError(_)));
    let good = image_scene(colored_image([220, 35, 10, 255]));
    let actual = gpu.render(&mut renderer, &mut resources, &good);
    assert_eq!(
        &actual[(110 * usize::from(SIZE) + 100) * 4..][..4],
        &[220, 35, 10, 255]
    );
}

#[test]
fn two_bitmap_scenes_encoded_before_submission_preserve_their_pixels() {
    let gpu = Gpu::new();
    let (mut renderer, mut resources) = gpu.renderer(AtlasConfig {
        atlas_size: (16, 16),
        max_atlases: 1,
        ..Default::default()
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut buffers = Vec::new();
    for color in [[220, 35, 10, 255], [10, 35, 220, 255]] {
        let scene = image_scene(colored_image(color));
        let target = gpu.texture();
        gpu.encode(&mut renderer, &mut resources, &scene, &target, &mut encoder)
            .unwrap();
        buffers.push((gpu.read(&target, &mut encoder), color));
    }
    gpu.queue.submit([encoder.finish()]);
    for (buffer, color) in buffers {
        let actual = gpu.pixels(&buffer);
        assert_eq!(&actual[(110 * usize::from(SIZE) + 100) * 4..][..4], &color);
        assert_eq!(bounds(&actual), [100, 110, 111, 121]);
    }
}

#[test]
fn adjacent_linear_sampled_pixmaps_match_cpu_without_color_bleeding() {
    let gpu = Gpu::new();
    let (mut renderer, mut resources) = gpu.renderer(AtlasConfig {
        atlas_size: (32, 32),
        max_atlases: 1,
        ..Default::default()
    });
    let images = [
        colored_image([180, 0, 0, 180]),
        colored_image([0, 0, 180, 180]),
    ];
    let mut scene = Scene::new(SIZE, SIZE);
    let mut cpu = vello_cpu::RenderContext::new(SIZE, SIZE);
    let mut cpu_resources = vello_cpu::Resources::new();
    for (index, image) in images.iter().enumerate() {
        let transform =
            Affine::translate((100.25 + index as f64 * 31.0, 110.3)) * Affine::scale(2.37);
        let paint = Image {
            image: ImageSource::Pixmap(Arc::clone(image)),
            sampler: ImageSampler {
                quality: peniko::ImageQuality::Medium,
                ..Default::default()
            },
        };
        scene.set_transform(transform);
        scene.set_paint(paint.clone());
        scene.fill_rect(&Rect::new(0.0, 0.0, 12.0, 12.0));
        cpu.set_transform(transform);
        cpu.set_paint(paint);
        cpu.fill_rect(&Rect::new(0.0, 0.0, 12.0, 12.0));
    }
    let mut reference = Pixmap::new(SIZE, SIZE);
    cpu.render(&mut reference, &mut cpu_resources);
    let actual = gpu.render(&mut renderer, &mut resources, &scene);
    compare(
        &actual,
        reference.data_as_u8_slice(),
        "adjacent-linear-pixmaps",
    );
    for (index, pixel) in actual.as_chunks::<4>().0.iter().enumerate() {
        let x = index % usize::from(SIZE);
        if x < 130 {
            assert_eq!(pixel[2], 0, "blue leaked into red image");
        }
        if x > 130 {
            assert_eq!(pixel[0], 0, "red leaked into blue image");
        }
    }
}

#[test]
fn empty_pixmap_writer_uploads_are_noops() {
    let gpu = Gpu::new();
    let target = gpu.texture();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for (width, height) in [(0, 12), (12, 0), (0, 0)] {
        let image = Pixmap::new(width, height);
        vello_gpu::AtlasWriter::write_to_atlas(
            &image,
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &target,
            [0, 0],
            width,
            height,
        );
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
fn ordinary_pixmap_upload_coexists_with_internally_submitted_colr_atlas_draws() {
    let gpu = Gpu::new();
    let (mut renderer, mut resources) = gpu.renderer(AtlasConfig::default());
    let bytes = include_bytes!("../../assets/noto_color_emoji/NotoColorEmoji-Subset.ttf").to_vec();
    let id = FontRef::from_index(&bytes, 0)
        .unwrap()
        .charmap()
        .map('✅')
        .unwrap()
        .to_u32();
    let font = peniko::FontData::new(peniko::Blob::new(Arc::new(bytes)), 0);
    for frame in 0..3 {
        let image = colored_image([220, 35, 10, 255]);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let uploaded = (frame == 1).then(|| {
            renderer.upload_image(
                &mut resources,
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &image,
            )
        });
        let font_size = 64.0 + frame as f32;
        let mut scene = glyph_scene(&mut resources, &font, id, true, font_size, Affine::IDENTITY);
        scene.set_transform(Affine::translate((100.0, 110.0)));
        scene.set_paint(Image {
            image: uploaded.map_or_else(
                || ImageSource::Pixmap(Arc::clone(&image)),
                ImageSource::opaque_id,
            ),
            sampler: ImageSampler::default(),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 12.0, 12.0));
        // Use the same cached-COLR raster path as an independent reference:
        // it differs slightly from uncached CPU rasterization at fractional
        // cache scales. The solid image occupies a disjoint, exact pixel region.
        let (mut reference_renderer, mut reference_resources) =
            gpu.renderer(AtlasConfig::default());
        let reference_scene = glyph_scene(
            &mut reference_resources,
            &font,
            id,
            true,
            font_size,
            Affine::IDENTITY,
        );
        let mut reference = gpu.render(
            &mut reference_renderer,
            &mut reference_resources,
            &reference_scene,
        );
        for y in 110..122 {
            for x in 100..112 {
                let offset = (y * usize::from(SIZE) + x) * 4;
                reference[offset..offset + 4].copy_from_slice(&[220, 35, 10, 255]);
            }
        }
        let target = gpu.texture();
        gpu.encode(&mut renderer, &mut resources, &scene, &target, &mut encoder)
            .unwrap();
        if let Some(id) = uploaded {
            renderer.destroy_image(&mut resources, &mut encoder, id);
        }
        let buffer = gpu.read(&target, &mut encoder);
        gpu.queue.submit([encoder.finish()]);
        compare(
            &gpu.pixels(&buffer),
            &reference,
            &format!("mixed-cached-colr-and-pixmap-{frame}"),
        );
    }
}

#[test]
fn direct_atlas_render_rejects_unprepared_pixmap_before_scheduling() {
    let gpu = Gpu::new();
    let config = AtlasConfig {
        atlas_size: (32, 32),
        max_atlases: 1,
        ..Default::default()
    };
    let (mut renderer, _) = gpu.renderer(config);
    let scene = image_scene(colored_image([220, 35, 10, 255]));
    let error = renderer
        .render_to_atlas(
            &scene,
            1,
            config,
            &gpu.device,
            &gpu.queue,
            vello_common::multi_atlas::AtlasId::new(0),
            &vello_gpu::TextureBindings::new(),
        )
        .unwrap_err();
    assert!(matches!(error, vello_gpu::RenderError::UnpreparedPixmap(_)));
}

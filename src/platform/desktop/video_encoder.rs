/// DEPRECATED BY `animation` MODULE (which can export frames with a constant framerate)
use std::{
    fs::File,
    io::Write,
    path::Path,
    process::{Child, Command, Stdio},
};

use sdl2::{pixels::PixelFormatEnum, rect::Rect, render::Canvas};

pub struct VideoEncoder {
    video_encoding_process: Child,
    width: u32,
    height: u32,
}
impl VideoEncoder {
    pub fn new(filename: &str, width: u32, height: u32) -> VideoEncoder {
        let video_encoding_process = Command::new("ffmpeg")
            .args(&[
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s:v",
                &format!("{}x{}", width, height),
                "-r",
                "120",
                "-i",
                "-",
                "-c:v",
                "libx264",
                filename,
                "-y",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to execute process");

        VideoEncoder {
            video_encoding_process,
            width,
            height,
        }
    }

    pub fn write(&mut self, b: &[u8]) {
        let video_encoding_target = self.video_encoding_process.stdin.as_mut().unwrap();
        video_encoding_target.write_all(&b).expect("failed to write pixels");
        video_encoding_target.flush().expect("failed to flush pixels");
    }

    pub fn write_canvas(&mut self, canvas: &Canvas<sdl2::video::Window>) {
        let mut pixels_vec = canvas
            .read_pixels(Rect::new(0, 0, self.width, self.height), PixelFormatEnum::BGR888)
            .expect("could not read pixels");
        for y in 0..800 {
            for x in 0..800 {
                // pixels_vec[4 * (y * 800 + x) + 0] = 255;
                // pixels_vec[4 * (y * 800 + x) + 1] = 255;
                // pixels_vec[4 * (y * 800 + x) + 2] = 255;
                pixels_vec[4 * (y * 800 + x) + 3] = 255;
            }
        }
        self.write(&pixels_vec);
    }
}

pub fn capture_foto(basepath: &str, canvas: &Canvas<sdl2::video::Window>) {
    let (width, height) = canvas.output_size().unwrap();
    let mut pixels_vec = canvas
        .read_pixels(Rect::new(0, 0, width, height), PixelFormatEnum::BGR888)
        .expect("could not read pixels");
    for y in 0..height as usize {
        for x in 0..width as usize {
            // pixels_vec[4 * (y * 800 + x) + 0] = 255;
            // pixels_vec[4 * (y * 800 + x) + 1] = 255;
            // pixels_vec[4 * (y * 800 + x) + 2] = 255;
            pixels_vec[4 * (y * width as usize + x) + 3] = 255;
        }
    }

    let mut i = 0;
    let filename = loop {
        let s = format!("{}-{}.png", basepath, i);
        if !Path::new(&s).exists() {
            break s;
        }
        i += 1;
    };
    let file = File::create(filename).unwrap();

    let mut encoder = png::Encoder::new(file, width, height); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_trns(vec![0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]);
    encoder.set_source_gamma(png::ScaledFloat::from_scaled(45455)); // 1.0 / 2.2, scaled by 100000
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2)); // 1.0 / 2.2, unscaled, but rounded
    let source_chromaticities = png::SourceChromaticities::new(
        // Using unscaled instantiation here
        (0.31270, 0.32900),
        (0.64000, 0.33000),
        (0.30000, 0.60000),
        (0.15000, 0.06000),
    );
    encoder.set_source_chromaticities(source_chromaticities);
    let mut writer = encoder.write_header().unwrap();

    // let data = [255, 0, 0, 255, 0, 0, 0, 255]; // An array containing a RGBA sequence. First pixel is red and second pixel is black.
    writer.write_image_data(&pixels_vec).unwrap(); // Save
}

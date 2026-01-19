import glfw
import moderngl
import numpy as np
import sounddevice as sd
from scipy.fft import rfft
import time
import random


WIDTH, HEIGHT = 1280, 720

if not glfw.init():
    raise RuntimeError("GLFW init failed")

window = glfw.create_window(WIDTH, HEIGHT, "Black Hole ", None, None)
glfw.make_context_current(window)

ctx = moderngl.create_context()


VERTEX_SHADER = """
#version 330
in vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
precision highp float;

uniform float t;
uniform vec2 r;
uniform float audioLevel;
uniform float bassLevel;
uniform float midLevel;
uniform float trebleLevel;

out vec4 fragColor;

vec2 myTanh(vec2 x) {
    vec2 ex = exp(x);
    vec2 emx = exp(-x);
    return (ex - emx) / (ex + emx);
}


float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec4 o_bg = vec4(0.0);
    vec4 o_anim = vec4(0.0);

    
    vec2 starCoord = gl_FragCoord.xy / r.y;
    vec2 starGrid = floor(starCoord * 100.0);
    float starHash = hash(starGrid);
    
    if (starHash > 0.98) {
        vec2 starPos = fract(starCoord * 100.0);
        float starDist = length(starPos - 0.5);
        if (starDist < 0.1) {
            float starBrightness = (1.0 - starDist * 10.0) * (0.3 + audioLevel * 0.7);
            o_bg += vec4(starBrightness);
        }
    }

    
    {
        vec2 p_img = (gl_FragCoord.xy * 2.0 - r) / r.y
                     * mat2(1.0, -1.0, 1.0, 1.0);

        vec2 l_val = myTanh(p_img * 5.0 + 2.0);
        l_val = min(l_val, l_val * 3.0);
        vec2 clamped = clamp(l_val, -2.0, 0.0);
        float diff_y = clamped.y - l_val.y;
        float safe_px = abs(p_img.x) < 0.001 ? 0.001 : p_img.x;

        float term = (0.1
            - max(0.01 - dot(p_img, p_img) / 200.0, 0.0)
            * (diff_y / safe_px))
            / abs(length(p_img) - 0.7);

        o_bg += vec4(term);
        o_bg *= max(o_bg, vec4(0.0));
    }

   
    {
        vec2 p_anim = (gl_FragCoord.xy * 2.0 - r) / r.y
                      / (0.7 + audioLevel * 0.3);

        vec2 d = vec2(-1.0, 1.0);
        float denom = 0.1 + 5.0 / dot(5.0 * p_anim - d, 5.0 * p_anim - d);
        vec2 c = p_anim * mat2(1.0, 1.0, d.x / denom, d.y / denom);
        vec2 v = c;

        float audioRotation = t * 0.2 + bassLevel * 5.0;
        v *= mat2(
            cos(log(length(v)) + audioRotation + vec4(0.0, 33.0, 11.0, 0.0))
        ) * (5.0 + midLevel * 3.0);

        vec4 animAccum = vec4(0.0);

        for (int i = 1; i <= 9; i++) {
            float fi = float(i);
            animAccum += sin(vec4(v.x, v.y, v.y, v.x)) + vec4(1.0);
            v += 0.7 * sin(vec2(v.y, v.x) * fi + t + trebleLevel * 3.0) / fi + 0.5;
        }

        vec4 animTerm =
            1.0 - exp(
                -exp(c.x * vec4(0.6, -0.4, -1.0, 0.0))
                / animAccum
                / (0.1 + 0.1 *
                   pow(length(sin(v / 0.3) * 0.2
                       + c * vec2(1.0, 2.0)) - 1.0, 2.0))
                / (1.0 + 7.0 * exp(0.3 * c.y - dot(c, c)))
                / (0.03 + abs(length(p_anim) - 0.7))
                * (0.2 + audioLevel * 0.3)
            );

        o_anim += animTerm;
    }

    vec4 finalColor = mix(o_bg, o_anim, 0.5)
                      * (1.5 + audioLevel * 0.8);

    fragColor = clamp(finalColor, 0.0, 1.0);
}
"""

prog = ctx.program(
    vertex_shader=VERTEX_SHADER,
    fragment_shader=FRAGMENT_SHADER
)


quad = np.array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1
], dtype='f4')

vbo = ctx.buffer(quad)
vao = ctx.simple_vertex_array(prog, vbo, 'a_position')


SAMPLE_RATE = 44100
FFT_SIZE = 2048  
audioLevel = bassLevel = midLevel = trebleLevel = 0.0
accumulated_time = 0.0
last_frame_time = time.time()


audio_smooth = 0.0
bass_smooth = 0.0
mid_smooth = 0.0
treble_smooth = 0.0
SMOOTHING = 0.3  
def audio_callback(indata, frames, time_info, status):
    global audioLevel, bassLevel, midLevel, trebleLevel
    global audio_smooth, bass_smooth, mid_smooth, treble_smooth
    
    if status:
        print(f"Audio status: {status}")
    
   
    samples = np.mean(indata, axis=1)
    
    
    window = np.hanning(len(samples))
    samples = samples * window
    
   
    fft = np.abs(rfft(samples))
    
   
    fft = np.log10(fft + 1) * 10
    
    
    
    bass_bins = fft[0:14]
    mid_bins = fft[14:105]
    treble_bins = fft[105:418]
    
    
    raw_audio = np.mean(fft) / 8.0
    raw_bass = np.mean(bass_bins) / 5.0
    raw_mid = np.mean(mid_bins) / 8.0
    raw_treble = np.mean(treble_bins) / 10.0
    
    
    audio_smooth = audio_smooth * (1 - SMOOTHING) + raw_audio * SMOOTHING
    bass_smooth = bass_smooth * (1 - SMOOTHING) + raw_bass * SMOOTHING
    mid_smooth = mid_smooth * (1 - SMOOTHING) + raw_mid * SMOOTHING
    treble_smooth = treble_smooth * (1 - SMOOTHING) + raw_treble * SMOOTHING
    
    
    audioLevel = min(audio_smooth, 1.0)
    bassLevel = min(bass_smooth, 1.0)
    midLevel = min(mid_smooth, 1.0)
    trebleLevel = min(treble_smooth, 1.0)


try:
   
    sd.default.device = "Stereo Mix"  
   
    print(f"Using audio device: {sd.query_devices(sd.default.device[0])['name']}")
except:
    print("Could not set system audio device. Using default microphone.")
    print("\nAvailable devices:")
    print(sd.query_devices())
    print("\nTo capture system audio:")
    print("Windows: Install VB-Cable and set device to 'CABLE Output' or enable 'Stereo Mix'")
    print("Mac: Install BlackHole and set device to 'BlackHole 2ch'")
    print("Linux: Use 'Monitor of [device]' or set up PulseAudio loopback")

stream = sd.InputStream(
    channels=2,
    samplerate=SAMPLE_RATE,
    blocksize=FFT_SIZE,
    callback=audio_callback
)
stream.start()

start = time.time()
frame_count = 0

print("\n=== Black Hole Music Visualizer ===")
print("Play some music and watch it react!")
print("Press ESC to exit\n")

while not glfw.window_should_close(window):
    glfw.poll_events()
    
    
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        break
    
    
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time
    
    
    time_multiplier = 1.0 + (audioLevel * 3.0) + (bassLevel * 5.0)
    accumulated_time += delta_time * time_multiplier
    
    
    frame_count += 1
    if frame_count % 60 == 0:
        print(f"Audio: {audioLevel:.3f} | Bass: {bassLevel:.3f} | Mid: {midLevel:.3f} | Treble: {trebleLevel:.3f} | Speed: {time_multiplier:.2f}x")
    
    
    prog['t'].value = accumulated_time
    prog['r'].value = (WIDTH, HEIGHT)
    prog['audioLevel'].value = audioLevel
    prog['bassLevel'].value = bassLevel
    prog['midLevel'].value = midLevel
    prog['trebleLevel'].value = trebleLevel

    ctx.clear()
    vao.render()
    glfw.swap_buffers(window)

stream.stop()
glfw.terminate()
print("\nVisualization closed. Goodbye!")

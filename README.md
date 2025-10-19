# Iq-to-WAV

 Convert raw interleaved IQ samples to 16-bit WAV (I -> left, Q -> right).
 You can use it for SDR# or other software.

 Requirements
 - Python 3.8+
 - numpy

 Quick install
 ```powershell
 python -m pip install numpy
 ```

 Usage
 ```powershell
 python main.py -i input.iq -o output.wav -r sample_rate -d type --gain 1.0
 ```

 Options
 -i  input .iq file
 -o  output .wav file
 -r  sample rate in Hz
 -d  input dtype (f32|i16|u8)
 --gain  linear gain
 --mono  mono (I only)

 Example
 ```powershell
 python main.py -i capture.iq -o out.wav -r 48000 -d f32 --gain 1.0
 ```

Reverse (WAV -> IQ)
```powershell
python main.py -i stereo.wav -o out.iq -d f32 --wav2iq
```
/* fsdd_frame.c -- framing pipeline for the Free Spoken Digit Dataset (real-audio direction).
 *
 * Turns the variable-length FSDD WAV clips into fixed-length, normalized, time-domain feature vectors
 * the emergence experiments can consume. Pure C99 + POSIX dirent, no audio libraries: a WAV file is a
 * 44-ish-byte header plus signed 16-bit little-endian PCM, so we parse it directly.
 *
 * Deliberately TIME-DOMAIN (raw waveform, no STFT/mel): the whole point is to test whether pitch/formant
 * filters EMERGE under the energy budget, which a spectral front-end would pre-impose.
 *
 * The reusable pieces are wav_read() and frame_clip(); main() is a validator that loads the whole set,
 * cross-checks the parse against known dataset facts (8 kHz mono 16-bit, 300 clips/digit), and reports a
 * per-digit autocorrelation pitch estimate to confirm the framing preserves voiced periodicity.
 *
 * Build: cc -std=c99 -pedantic -Wall -Wextra -O2 fsdd_frame.c -lm -o fsdd_frame
 * Run:   ./fsdd_frame [N] [recordings_dir]      (defaults: N=256, data/fsdd/recordings)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dirent.h>

#define MAXRAW  8000     /* max raw samples read per clip (1 s at 8 kHz; clips are ~0.25-0.5 s) */
#define MAXN    1024     /* max framed length */

/* Read a MONO 16-bit PCM WAV at `path` into out[] normalized to [-1,1]; returns #samples (<=maxn),
 * or -1 on error. Scans RIFF chunks for "fmt " and "data" instead of assuming a fixed 44-byte header,
 * so files with extra chunks still parse. Reports sample rate / channels / bits via the out-params. */
static int wav_read(const char *path, double *out, int maxn, int *rate, int *chans, int *bits)
{
    unsigned char hdr[12], ck[8];
    int c = 0, b = 0, sr = 0, n = 0, got_fmt = 0;
    long datasz = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) { fclose(f); return -1; }
    while (fread(ck, 1, 8, f) == 8) {
        uint32_t sz = (uint32_t)ck[4] | (uint32_t)ck[5] << 8 | (uint32_t)ck[6] << 16 | (uint32_t)ck[7] << 24;
        if (!memcmp(ck, "fmt ", 4)) {
            unsigned char fb[16];
            if (sz < 16 || fread(fb, 1, 16, f) != 16) { fclose(f); return -1; }
            c  = (int)(fb[2] | fb[3] << 8);
            sr = (int)((uint32_t)fb[4] | (uint32_t)fb[5] << 8 | (uint32_t)fb[6] << 16 | (uint32_t)fb[7] << 24);
            b  = (int)(fb[14] | fb[15] << 8);
            if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
            got_fmt = 1;
        } else if (!memcmp(ck, "data", 4)) {
            datasz = (long)sz;
            break;                                   /* PCM samples follow immediately */
        } else {
            fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);   /* skip unknown chunk (RIFF pads to even) */
        }
    }
    if (!got_fmt || datasz <= 0 || c != 1 || b != 16) { fclose(f); return -1; }
    { long ns = datasz / 2, i;
      for (i = 0; i < ns && n < maxn; i++) {
          unsigned char s2[2];
          if (fread(s2, 1, 2, f) != 2) break;
          out[n++] = (double)(int16_t)(s2[0] | s2[1] << 8) / 32768.0;
      } }
    fclose(f);
    if (rate)  *rate  = sr;
    if (chans) *chans = c;
    if (bits)  *bits  = b;
    return n;
}

/* Frame a raw clip raw[len] into a fixed N-sample window x[N]: take the maximum-energy N-window (the
 * loudest voiced region), or zero-pad centered if the clip is shorter than N; then normalize the window
 * to zero mean and unit max-abs so amplitude is consistent across speakers. */
static void frame_clip(const double *raw, int len, double *x, int N)
{
    int i;
    if (len <= N) {
        int pad = (N - len) / 2;
        for (i = 0; i < N; i++) x[i] = 0.0;
        for (i = 0; i < len; i++) x[pad + i] = raw[i];
    } else {
        int step = N / 8 > 0 ? N / 8 : 1, s, best = 0;
        double bestE = -1.0;
        for (s = 0; s + N <= len; s += step) {
            double e = 0.0;
            for (i = 0; i < N; i++) e += raw[s + i] * raw[s + i];
            if (e > bestE) { bestE = e; best = s; }
        }
        for (i = 0; i < N; i++) x[i] = raw[best + i];
    }
    { double m = 0.0, mx = 1e-9;
      for (i = 0; i < N; i++) m += x[i];
      m /= N;
      for (i = 0; i < N; i++) { x[i] -= m; if (fabs(x[i]) > mx) mx = fabs(x[i]); }
      for (i = 0; i < N; i++) x[i] /= mx; }
}

/* Dominant autocorrelation lag of x[N] within [lo,hi] -- a crude pitch-period estimate (samples). */
static int pitch_lag(const double *x, int N, int lo, int hi)
{
    int L, best = lo;
    double bestR = -1e18;
    for (L = lo; L <= hi && L < N; L++) {
        double r = 0.0;
        int i;
        for (i = 0; i + L < N; i++) r += x[i] * x[i + L];
        if (r > bestR) { bestR = r; best = L; }
    }
    return best;
}

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 256;
    const char *dir = argc > 2 ? argv[2] : "/home/vas/smbpann/data/fsdd/recordings";
    double raw[MAXRAW], x[MAXN];
    int count[10] = {0}, i, badrate = 0, badfmt = 0, nfiles = 0;
    long sumlen = 0;
    int minlen = 1 << 30, maxlen = 0;
    double lagsum[10] = {0};
    int firstrate = 0, firstchans = 0, firstbits = 0, printed_first = 0;
    DIR *d;
    struct dirent *e;

    if (N < 16 || N > MAXN) { fprintf(stderr, "N out of range [16,%d]\n", MAXN); return 2; }
    d = opendir(dir);
    if (!d) { fprintf(stderr, "cannot open dir: %s\n", dir); return 1; }

    printf("FSDD framing pipeline: N=%d, peak-energy window @ native 8 kHz, per-clip normalized\n", N);
    printf("dir=%s\n\n", dir);

    while ((e = readdir(d))) {
        char path[1024];
        int len, rate = 0, ch = 0, bits = 0, digit;
        const char *nm = e->d_name;
        size_t L = strlen(nm);
        if (L < 5 || strcmp(nm + L - 4, ".wav") || nm[0] < '0' || nm[0] > '9') continue;
        digit = nm[0] - '0';
        snprintf(path, sizeof path, "%s/%s", dir, nm);
        len = wav_read(path, raw, MAXRAW, &rate, &ch, &bits);
        if (len < 0) { badfmt++; continue; }
        if (!printed_first) { firstrate = rate; firstchans = ch; firstbits = bits; printed_first = 1; }
        if (rate != 8000 || ch != 1 || bits != 16) badrate++;
        frame_clip(raw, len, x, N);
        count[digit]++;
        nfiles++;
        sumlen += len;
        if (len < minlen) minlen = len;
        if (len > maxlen) maxlen = len;
        lagsum[digit] += pitch_lag(x, N, 20, 160);   /* 8 kHz: 20-160 samples ~ 50-400 Hz pitch */
    }
    closedir(d);

    if (nfiles == 0) { fprintf(stderr, "no .wav files parsed under %s\n", dir); return 1; }

    printf("parsed %d clips  (format errors: %d)\n", nfiles, badfmt);
    printf("first-file parse: rate=%d Hz, channels=%d, bits=%d   (expect 8000 / 1 / 16)\n",
           firstrate, firstchans, firstbits);
    printf("clips off-spec (not 8 kHz mono 16-bit): %d\n", badrate);
    printf("raw length: min=%d max=%d mean=%.0f samples (%.2f s mean @ 8 kHz)\n\n",
           minlen, maxlen, (double)sumlen / nfiles, (double)sumlen / nfiles / 8000.0);

    printf("per-digit clip count (expect 300 each) and mean pitch-lag (samples; voiced ~ 30-90):\n");
    for (i = 0; i < 10; i++)
        printf("  digit %d : %3d clips   mean pitch-lag %.1f  (~%.0f Hz)\n",
               i, count[i], count[i] ? lagsum[i] / count[i] : 0.0,
               count[i] && lagsum[i] > 0 ? 8000.0 / (lagsum[i] / count[i]) : 0.0);
    return 0;
}

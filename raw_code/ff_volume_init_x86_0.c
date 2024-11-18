av_cold void ff_volume_init_x86(VolumeContext *vol)

{

    int cpu_flags = av_get_cpu_flags();

    enum AVSampleFormat sample_fmt = av_get_packed_sample_fmt(vol->sample_fmt);



    if (sample_fmt == AV_SAMPLE_FMT_S16) {

        if (EXTERNAL_SSE2(cpu_flags) && vol->volume_i < 32768) {

            vol->scale_samples = ff_scale_samples_s16_sse2;

            vol->samples_align = 8;

        }

    } else if (sample_fmt == AV_SAMPLE_FMT_S32) {

        if (EXTERNAL_SSE2(cpu_flags)) {

            vol->scale_samples = ff_scale_samples_s32_sse2;

            vol->samples_align = 4;

        }

        if (EXTERNAL_SSSE3(cpu_flags) && cpu_flags & AV_CPU_FLAG_ATOM) {

            vol->scale_samples = ff_scale_samples_s32_ssse3_atom;

            vol->samples_align = 4;

        }

        if (EXTERNAL_AVX(cpu_flags)) {

            vol->scale_samples = ff_scale_samples_s32_avx;

            vol->samples_align = 8;

        }

    }

}

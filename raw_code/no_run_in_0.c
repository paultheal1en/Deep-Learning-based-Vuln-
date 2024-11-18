static int no_run_in (HWVoiceIn *hw)

{

    NoVoiceIn *no = (NoVoiceIn *) hw;

    int live = audio_pcm_hw_get_live_in (hw);

    int dead = hw->samples - live;

    int samples = 0;



    if (dead) {

        int64_t now = qemu_get_clock (vm_clock);

        int64_t ticks = now - no->old_ticks;

        int64_t bytes =

            muldiv64 (ticks, hw->info.bytes_per_second, get_ticks_per_sec ());



        no->old_ticks = now;

        bytes = audio_MIN (bytes, INT_MAX);

        samples = bytes >> hw->info.shift;

        samples = audio_MIN (samples, dead);

    }

    return samples;

}

int pit_get_out(PITState *pit, int channel, int64_t current_time)

{

    PITChannelState *s = &pit->channels[channel];

    return pit_get_out1(s, current_time);

}

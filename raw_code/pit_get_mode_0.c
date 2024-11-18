int pit_get_mode(PITState *pit, int channel)

{

    PITChannelState *s = &pit->channels[channel];

    return s->mode;

}

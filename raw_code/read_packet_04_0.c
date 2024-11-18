static int read_packet(AVFormatContext* ctx, AVPacket *pkt)

{

    al_data *ad = ctx->priv_data;

    int error=0;

    const char *error_msg;

    ALCint nb_samples;



    /* Get number of samples available */

    alcGetIntegerv(ad->device, ALC_CAPTURE_SAMPLES, (ALCsizei) sizeof(ALCint), &nb_samples);

    if (error = al_get_error(ad->device, &error_msg)) goto fail;



    /* Create a packet of appropriate size */

    av_new_packet(pkt, nb_samples*ad->sample_step);

    pkt->pts = av_gettime();



    /* Fill the packet with the available samples */

    alcCaptureSamples(ad->device, pkt->data, nb_samples);

    if (error = al_get_error(ad->device, &error_msg)) goto fail;



    return pkt->size;

fail:

    /* Handle failure */

    if (pkt->data)

        av_destruct_packet(pkt);

    if (error_msg)

        av_log(ctx, AV_LOG_ERROR, "Error: %s\n", error_msg);

    return error;

}

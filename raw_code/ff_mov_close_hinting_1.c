void ff_mov_close_hinting(MOVTrack *track) {

    AVFormatContext* rtp_ctx = track->rtp_ctx;

    uint8_t *ptr;



    av_freep(&track->enc);

    sample_queue_free(&track->sample_queue);

    if (!rtp_ctx)

        return;

    if (rtp_ctx->pb) {

        av_write_trailer(rtp_ctx);

        url_close_dyn_buf(rtp_ctx->pb, &ptr);

        av_free(ptr);

    }

    av_metadata_free(&rtp_ctx->streams[0]->metadata);

    av_metadata_free(&rtp_ctx->metadata);


    av_free(rtp_ctx->streams[0]);

    av_freep(&rtp_ctx);

}
static int context_init(H264Context *h){

    MpegEncContext * const s = &h->s;



    CHECKED_ALLOCZ(h->top_borders[0], h->s.mb_width * (16+8+8) * sizeof(uint8_t))

    CHECKED_ALLOCZ(h->top_borders[1], h->s.mb_width * (16+8+8) * sizeof(uint8_t))



    // edge emu needs blocksize + filter length - 1 (=17x17 for halfpel / 21x21 for h264)

    CHECKED_ALLOCZ(s->allocated_edge_emu_buffer,

                   (s->width+64)*2*21*2); //(width + edge + align)*interlaced*MBsize*tolerance

    s->edge_emu_buffer= s->allocated_edge_emu_buffer + (s->width+64)*2*21;

    return 0;

fail:

    return -1; // free_tables will clean up for us

}

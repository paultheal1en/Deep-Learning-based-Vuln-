static void x8_init_block_index(MpegEncContext *s){ //FIXME maybe merge with ff_*

//not s->linesize as this would be wrong for field pics

//not that IntraX8 has interlacing support ;)

    const int linesize   = s->current_picture.f.linesize[0];

    const int uvlinesize = s->current_picture.f.linesize[1];



    s->dest[0] = s->current_picture.f.data[0];

    s->dest[1] = s->current_picture.f.data[1];

    s->dest[2] = s->current_picture.f.data[2];



    s->dest[0] +=   s->mb_y        *   linesize << 3;

    s->dest[1] += ( s->mb_y&(~1) ) * uvlinesize << 2;//chroma blocks are on add rows

    s->dest[2] += ( s->mb_y&(~1) ) * uvlinesize << 2;

}

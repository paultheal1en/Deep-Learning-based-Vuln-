int avcodec_default_get_buffer(AVCodecContext *s, AVFrame *pic){
    int i;
    int w= s->width;
    int h= s->height;
    InternalBuffer *buf;
    int *picture_number;
    if(pic->data[0]!=NULL) {
        av_log(s, AV_LOG_ERROR, "pic->data[0]!=NULL in avcodec_default_get_buffer\n");
        return -1;
    }
    if(s->internal_buffer_count >= INTERNAL_BUFFER_SIZE) {
        av_log(s, AV_LOG_ERROR, "internal_buffer_count overflow (missing release_buffer?)\n");
        return -1;
    }
    if(avcodec_check_dimensions(s,w,h))
        return -1;
    if(s->internal_buffer==NULL){
        s->internal_buffer= av_mallocz(INTERNAL_BUFFER_SIZE*sizeof(InternalBuffer));
    }
#if 0
    s->internal_buffer= av_fast_realloc(
        s->internal_buffer,
        &s->internal_buffer_size,
        sizeof(InternalBuffer)*FFMAX(99,  s->internal_buffer_count+1)/*FIXME*/
        );
#endif
    buf= &((InternalBuffer*)s->internal_buffer)[s->internal_buffer_count];
    picture_number= &(((InternalBuffer*)s->internal_buffer)[INTERNAL_BUFFER_SIZE-1]).last_pic_num; //FIXME ugly hack
    (*picture_number)++;
    if(buf->base[0]){
        pic->age= *picture_number - buf->last_pic_num;
        buf->last_pic_num= *picture_number;
    }else{
        int h_chroma_shift, v_chroma_shift;
        int pixel_size, size[3];
        AVPicture picture;
        avcodec_get_chroma_sub_sample(s->pix_fmt, &h_chroma_shift, &v_chroma_shift);
        if(!(s->flags&CODEC_FLAG_EMU_EDGE)){
            w+= EDGE_WIDTH*2;
            h+= EDGE_WIDTH*2;
        }
        avpicture_fill(&picture, NULL, s->pix_fmt, w, h);
        pixel_size= picture.linesize[0]*8 / w;
//av_log(NULL, AV_LOG_ERROR, "%d %d %d %d\n", (int)picture.data[1], w, h, s->pix_fmt);
        assert(pixel_size>=1);
            //FIXME next ensures that linesize= 2^x uvlinesize, thats needed because some MC code assumes it
        if(pixel_size == 3*8)
            w= ALIGN(w, STRIDE_ALIGN<<h_chroma_shift);
        else
            w= ALIGN(pixel_size*w, STRIDE_ALIGN<<(h_chroma_shift+3)) / pixel_size;
        size[1] = avpicture_fill(&picture, NULL, s->pix_fmt, w, h);
        size[0] = picture.linesize[0] * h;
        size[1] -= size[0];
        if(picture.data[2])
            size[1]= size[2]= size[1]/2;
        else
            size[2]= 0;
        buf->last_pic_num= -256*256*256*64;
        memset(buf->base, 0, sizeof(buf->base));
        memset(buf->data, 0, sizeof(buf->data));
        for(i=0; i<3 && size[i]; i++){
            const int h_shift= i==0 ? 0 : h_chroma_shift;
            const int v_shift= i==0 ? 0 : v_chroma_shift;
            buf->linesize[i]= picture.linesize[i];
            buf->base[i]= av_malloc(size[i]+16); //FIXME 16
            if(buf->base[i]==NULL) return -1;
            memset(buf->base[i], 128, size[i]);
            // no edge if EDEG EMU or not planar YUV, we check for PAL8 redundantly to protect against a exploitable bug regression ...
            if((s->flags&CODEC_FLAG_EMU_EDGE) || (s->pix_fmt == PIX_FMT_PAL8) || !size[2])
                buf->data[i] = buf->base[i];
            else
                buf->data[i] = buf->base[i] + ALIGN((buf->linesize[i]*EDGE_WIDTH>>v_shift) + (EDGE_WIDTH>>h_shift), STRIDE_ALIGN);
        }
        pic->age= 256*256*256*64;
    }
    pic->type= FF_BUFFER_TYPE_INTERNAL;
    for(i=0; i<4; i++){
        pic->base[i]= buf->base[i];
        pic->data[i]= buf->data[i];
        pic->linesize[i]= buf->linesize[i];
    }
    s->internal_buffer_count++;
    return 0;
}
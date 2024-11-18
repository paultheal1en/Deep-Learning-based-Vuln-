static int vmd_read_header(AVFormatContext *s,

                           AVFormatParameters *ap)

{

    VmdDemuxContext *vmd = (VmdDemuxContext *)s->priv_data;

    ByteIOContext *pb = &s->pb;

    AVStream *st;

    unsigned int toc_offset;

    unsigned char *raw_frame_table;

    int raw_frame_table_size;

    offset_t current_offset;

    int i, j;

    unsigned int total_frames;

    int64_t video_pts_inc = 0;

    int64_t current_video_pts = 0;

    unsigned char chunk[BYTES_PER_FRAME_RECORD];

    int lastframe = 0;



    /* fetch the main header, including the 2 header length bytes */

    url_fseek(pb, 0, SEEK_SET);

    if (get_buffer(pb, vmd->vmd_header, VMD_HEADER_SIZE) != VMD_HEADER_SIZE)

        return AVERROR_IO;



    vmd->audio_sample_counter = 0;

    vmd->audio_frame_divisor = 1;

    vmd->audio_block_align = 1;



    /* start up the decoders */

    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR_NOMEM;

    av_set_pts_info(st, 33, 1, 90000);

    vmd->video_stream_index = st->index;

    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_VMDVIDEO;

    st->codec->codec_tag = 0;  /* no fourcc */

    st->codec->width = LE_16(&vmd->vmd_header[12]);

    st->codec->height = LE_16(&vmd->vmd_header[14]);

    st->codec->time_base.num = 1;

    st->codec->time_base.den = 10;

    st->codec->extradata_size = VMD_HEADER_SIZE;

    st->codec->extradata = av_mallocz(VMD_HEADER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);

    memcpy(st->codec->extradata, vmd->vmd_header, VMD_HEADER_SIZE);



    /* if sample rate is 0, assume no audio */

    vmd->sample_rate = LE_16(&vmd->vmd_header[804]);

    if (vmd->sample_rate) {

        st = av_new_stream(s, 0);

        if (!st)

            return AVERROR_NOMEM;

        av_set_pts_info(st, 33, 1, 90000);

        vmd->audio_stream_index = st->index;

        st->codec->codec_type = CODEC_TYPE_AUDIO;

        st->codec->codec_id = CODEC_ID_VMDAUDIO;

        st->codec->codec_tag = 0;  /* no fourcc */

        st->codec->channels = vmd->audio_channels = (vmd->vmd_header[811] & 0x80) ? 2 : 1;

        st->codec->sample_rate = vmd->sample_rate;

        st->codec->block_align = vmd->audio_block_align =

            LE_16(&vmd->vmd_header[806]);

        if (st->codec->block_align & 0x8000) {

            st->codec->bits_per_sample = 16;

            st->codec->block_align = -(st->codec->block_align - 0x10000);

            vmd->audio_block_align = -(vmd->audio_block_align - 0x10000);

        } else {

            st->codec->bits_per_sample = 8;


        st->codec->bit_rate = st->codec->sample_rate *

            st->codec->bits_per_sample * st->codec->channels;



        /* for calculating pts */

        vmd->audio_frame_divisor = st->codec->channels;



        video_pts_inc = 90000;

        video_pts_inc *= st->codec->block_align;

        video_pts_inc /= st->codec->sample_rate;

        video_pts_inc /= st->codec->channels;

    } else {

        /* if no audio, assume 10 frames/second */

        video_pts_inc = 90000 / 10;




    toc_offset = LE_32(&vmd->vmd_header[812]);

    vmd->frame_count = LE_16(&vmd->vmd_header[6]);

    vmd->frames_per_block = LE_16(&vmd->vmd_header[18]);

    url_fseek(pb, toc_offset, SEEK_SET);



    raw_frame_table = NULL;

    vmd->frame_table = NULL;

    raw_frame_table_size = vmd->frame_count * 6;

    raw_frame_table = av_malloc(raw_frame_table_size);





    vmd->frame_table = av_malloc(vmd->frame_count * vmd->frames_per_block * sizeof(vmd_frame_t));

    if (!raw_frame_table || !vmd->frame_table) {

        av_free(raw_frame_table);

        av_free(vmd->frame_table);

        return AVERROR_NOMEM;


    if (get_buffer(pb, raw_frame_table, raw_frame_table_size) !=

        raw_frame_table_size) {

        av_free(raw_frame_table);

        av_free(vmd->frame_table);

        return AVERROR_IO;




    total_frames = 0;

    for (i = 0; i < vmd->frame_count; i++) {



        current_offset = LE_32(&raw_frame_table[6 * i + 2]);



        /* handle each entry in index block */

        for (j = 0; j < vmd->frames_per_block; j++) {

            int type;

            uint32_t size;



            get_buffer(pb, chunk, BYTES_PER_FRAME_RECORD);

            type = chunk[0];

            size = LE_32(&chunk[2]);

            if(!size)

                continue;

            switch(type) {

            case 1: /* Audio Chunk */

                vmd->frame_table[total_frames].frame_offset = current_offset;

                vmd->frame_table[total_frames].stream_index = vmd->audio_stream_index;

                vmd->frame_table[total_frames].frame_size = size;

                memcpy(vmd->frame_table[total_frames].frame_record, chunk, BYTES_PER_FRAME_RECORD);

                total_frames++;

                break;

            case 2: /* Video Chunk */

                vmd->frame_table[total_frames].frame_offset = current_offset;

                vmd->frame_table[total_frames].frame_size = size;

                vmd->frame_table[total_frames].stream_index = vmd->video_stream_index;

                memcpy(vmd->frame_table[total_frames].frame_record, chunk, BYTES_PER_FRAME_RECORD);

                vmd->frame_table[total_frames].pts = current_video_pts;

                if (lastframe) {

                    vmd->frame_table[lastframe].pts = current_video_pts - video_pts_inc;


                lastframe = total_frames;

                total_frames++;

                break;


            current_offset += size;


        current_video_pts += video_pts_inc;




    av_free(raw_frame_table);



    vmd->current_frame = 0;

    vmd->frame_count = total_frames;



    return 0;

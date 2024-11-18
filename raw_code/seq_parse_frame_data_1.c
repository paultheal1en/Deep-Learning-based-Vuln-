static int seq_parse_frame_data(SeqDemuxContext *seq, ByteIOContext *pb)

{

    unsigned int offset_table[4], buffer_num[4];

    TiertexSeqFrameBuffer *seq_buffer;

    int i, e;



    seq->current_frame_offs += SEQ_FRAME_SIZE;

    url_fseek(pb, seq->current_frame_offs, SEEK_SET);



    /* sound data */

    seq->current_audio_data_offs = get_le16(pb);

    if (seq->current_audio_data_offs != 0) {

        seq->current_audio_data_size = SEQ_AUDIO_BUFFER_SIZE * 2;

    } else {

        seq->current_audio_data_size = 0;

    }



    /* palette data */

    seq->current_pal_data_offs = get_le16(pb);

    if (seq->current_pal_data_offs != 0) {

        seq->current_pal_data_size = 768;

    } else {

        seq->current_pal_data_size = 0;

    }



    /* video data */

    for (i = 0; i < 4; i++)

        buffer_num[i] = get_byte(pb);



    for (i = 0; i < 4; i++)

        offset_table[i] = get_le16(pb);



    for (i = 0; i < 3; i++) {

        if (offset_table[i] != 0) {

            for (e = i + 1; e < 4 && offset_table[e] == 0; e++);

            seq_fill_buffer(seq, pb, buffer_num[1 + i],

              offset_table[i],

              offset_table[e] - offset_table[i]);

        }

    }



    if (buffer_num[0] != 255) {

        seq_buffer = &seq->frame_buffers[buffer_num[0]];

        seq->current_video_data_size = seq_buffer->fill_size;

        seq->current_video_data_ptr  = seq_buffer->data;

        seq_buffer->fill_size = 0;

    } else {

        seq->current_video_data_size = 0;

        seq->current_video_data_ptr  = 0;

    }



    return 0;

}

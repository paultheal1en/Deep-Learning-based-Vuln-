static int seq_fill_buffer(SeqDemuxContext *seq, ByteIOContext *pb, int buffer_num, unsigned int data_offs, int data_size)

{

    TiertexSeqFrameBuffer *seq_buffer;



    if (buffer_num >= SEQ_NUM_FRAME_BUFFERS)

        return AVERROR_INVALIDDATA;



    seq_buffer = &seq->frame_buffers[buffer_num];

    if (seq_buffer->fill_size + data_size > seq_buffer->data_size)

        return AVERROR_INVALIDDATA;



    url_fseek(pb, seq->current_frame_offs + data_offs, SEEK_SET);

    if (get_buffer(pb, seq_buffer->data + seq_buffer->fill_size, data_size) != data_size)

        return AVERROR(EIO);



    seq_buffer->fill_size += data_size;

    return 0;

}

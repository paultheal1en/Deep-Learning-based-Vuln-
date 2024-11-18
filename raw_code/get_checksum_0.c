unsigned long get_checksum(ByteIOContext *s){

    s->checksum= s->update_checksum(s->checksum, s->checksum_ptr, s->buf_ptr - s->checksum_ptr);

    s->checksum_ptr= NULL;

    return s->checksum;

}

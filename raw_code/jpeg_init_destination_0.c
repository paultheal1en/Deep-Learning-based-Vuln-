static void jpeg_init_destination(j_compress_ptr cinfo)

{

    VncState *vs = cinfo->client_data;

    Buffer *buffer = &vs->tight_jpeg;



    cinfo->dest->next_output_byte = (JOCTET *)buffer->buffer + buffer->offset;

    cinfo->dest->free_in_buffer = (size_t)(buffer->capacity - buffer->offset);

}

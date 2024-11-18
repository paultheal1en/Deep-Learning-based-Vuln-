void ff_slice_buffer_init(slice_buffer *buf, int line_count,

                          int max_allocated_lines, int line_width,

                          IDWTELEM *base_buffer)

{

    int i;



    buf->base_buffer = base_buffer;

    buf->line_count  = line_count;

    buf->line_width  = line_width;

    buf->data_count  = max_allocated_lines;

    buf->line        = av_mallocz(sizeof(IDWTELEM *) * line_count);

    buf->data_stack  = av_malloc(sizeof(IDWTELEM *) * max_allocated_lines);



    for (i = 0; i < max_allocated_lines; i++)

        buf->data_stack[i] = av_malloc(sizeof(IDWTELEM) * line_width);



    buf->data_stack_top = max_allocated_lines - 1;

}

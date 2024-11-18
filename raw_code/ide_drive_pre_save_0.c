static void ide_drive_pre_save(void *opaque)

{

    IDEState *s = opaque;



    s->cur_io_buffer_len = 0;



    if (!(s->status & DRQ_STAT))

        return;



    s->cur_io_buffer_offset = s->data_ptr - s->io_buffer;

    s->cur_io_buffer_len = s->data_end - s->data_ptr;



    s->end_transfer_fn_idx = transfer_end_table_idx(s->end_transfer_func);

    if (s->end_transfer_fn_idx == -1) {

        fprintf(stderr, "%s: invalid end_transfer_func for DRQ_STAT\n",

                        __func__);

        s->end_transfer_fn_idx = 2;

    }

}

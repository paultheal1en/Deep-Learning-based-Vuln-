static void qemu_fflush(QEMUFile *f)

{

    int ret = 0;



    if (!f->ops->put_buffer) {

        return;

    }

    if (f->is_write && f->buf_index > 0) {

        ret = f->ops->put_buffer(f->opaque, f->buf, f->buf_offset, f->buf_index);

        if (ret >= 0) {

            f->buf_offset += f->buf_index;

        }

        f->buf_index = 0;

    }

    if (ret < 0) {

        qemu_file_set_error(f, ret);

    }

}

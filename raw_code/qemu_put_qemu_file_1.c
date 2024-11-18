int qemu_put_qemu_file(QEMUFile *f_des, QEMUFile *f_src)

{

    int len = 0;



    if (f_src->buf_index > 0) {

        len = f_src->buf_index;

        qemu_put_buffer(f_des, f_src->buf, f_src->buf_index);

        f_src->buf_index = 0;


    }

    return len;

}
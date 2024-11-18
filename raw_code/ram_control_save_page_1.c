size_t ram_control_save_page(QEMUFile *f, ram_addr_t block_offset,

                         ram_addr_t offset, size_t size, int *bytes_sent)

{

    if (f->ops->save_page) {

        int ret = f->ops->save_page(f, f->opaque, block_offset,

                                    offset, size, bytes_sent);



        if (ret != RAM_SAVE_CONTROL_DELAYED) {

            if (*bytes_sent > 0) {

                qemu_update_position(f, *bytes_sent);

            } else if (ret < 0) {

                qemu_file_set_error(f, ret);

            }

        }



        return ret;

    }



    return RAM_SAVE_CONTROL_NOT_SUPP;

}

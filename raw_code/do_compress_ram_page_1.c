static int do_compress_ram_page(QEMUFile *f, RAMBlock *block,

                                ram_addr_t offset)

{

    RAMState *rs = &ram_state;

    int bytes_sent, blen;

    uint8_t *p = block->host + (offset & TARGET_PAGE_MASK);



    bytes_sent = save_page_header(rs, block, offset |

                                  RAM_SAVE_FLAG_COMPRESS_PAGE);

    blen = qemu_put_compression_data(f, p, TARGET_PAGE_SIZE,

                                     migrate_compress_level());

    if (blen < 0) {

        bytes_sent = 0;

        qemu_file_set_error(migrate_get_current()->to_dst_file, blen);

        error_report("compressed data failed!");

    } else {

        bytes_sent += blen;

        ram_release_pages(block->idstr, offset & TARGET_PAGE_MASK, 1);

    }



    return bytes_sent;

}

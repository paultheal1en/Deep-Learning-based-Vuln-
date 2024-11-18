static int ram_save_compressed_page(QEMUFile *f, PageSearchStatus *pss,

                                    bool last_stage,

                                    uint64_t *bytes_transferred)

{

    int pages = -1;

    uint64_t bytes_xmit;

    uint8_t *p;

    int ret;

    RAMBlock *block = pss->block;

    ram_addr_t offset = pss->offset;



    p = block->host + offset;



    bytes_xmit = 0;

    ret = ram_control_save_page(f, block->offset,

                                offset, TARGET_PAGE_SIZE, &bytes_xmit);

    if (bytes_xmit) {

        *bytes_transferred += bytes_xmit;

        pages = 1;

    }

    if (block == last_sent_block) {

        offset |= RAM_SAVE_FLAG_CONTINUE;

    }

    if (ret != RAM_SAVE_CONTROL_NOT_SUPP) {

        if (ret != RAM_SAVE_CONTROL_DELAYED) {

            if (bytes_xmit > 0) {

                acct_info.norm_pages++;

            } else if (bytes_xmit == 0) {

                acct_info.dup_pages++;

            }

        }

    } else {

        /* When starting the process of a new block, the first page of

         * the block should be sent out before other pages in the same

         * block, and all the pages in last block should have been sent

         * out, keeping this order is important, because the 'cont' flag

         * is used to avoid resending the block name.

         */

        if (block != last_sent_block) {

            flush_compressed_data(f);

            pages = save_zero_page(f, block, offset, p, bytes_transferred);

            if (pages == -1) {

                set_compress_params(&comp_param[0], block, offset);

                /* Use the qemu thread to compress the data to make sure the

                 * first page is sent out before other pages

                 */

                bytes_xmit = do_compress_ram_page(&comp_param[0]);

                acct_info.norm_pages++;

                qemu_put_qemu_file(f, comp_param[0].file);

                *bytes_transferred += bytes_xmit;

                pages = 1;

            }

        } else {

            pages = save_zero_page(f, block, offset, p, bytes_transferred);

            if (pages == -1) {

                pages = compress_page_with_multi_thread(f, block, offset,

                                                        bytes_transferred);

            }

        }

    }



    return pages;

}

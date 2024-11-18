static int ram_save_compressed_page(RAMState *rs, PageSearchStatus *pss,

                                    bool last_stage)

{

    int pages = -1;

    uint64_t bytes_xmit = 0;

    uint8_t *p;

    int ret, blen;

    RAMBlock *block = pss->block;

    ram_addr_t offset = pss->page << TARGET_PAGE_BITS;



    p = block->host + offset;



    ret = ram_control_save_page(rs->f, block->offset,

                                offset, TARGET_PAGE_SIZE, &bytes_xmit);

    if (bytes_xmit) {

        rs->bytes_transferred += bytes_xmit;

        pages = 1;

    }

    if (ret != RAM_SAVE_CONTROL_NOT_SUPP) {

        if (ret != RAM_SAVE_CONTROL_DELAYED) {

            if (bytes_xmit > 0) {

                rs->norm_pages++;

            } else if (bytes_xmit == 0) {

                rs->zero_pages++;

            }

        }

    } else {

        /* When starting the process of a new block, the first page of

         * the block should be sent out before other pages in the same

         * block, and all the pages in last block should have been sent

         * out, keeping this order is important, because the 'cont' flag

         * is used to avoid resending the block name.

         */

        if (block != rs->last_sent_block) {

            flush_compressed_data(rs);

            pages = save_zero_page(rs, block, offset, p);

            if (pages == -1) {

                /* Make sure the first page is sent out before other pages */

                bytes_xmit = save_page_header(rs, block, offset |

                                              RAM_SAVE_FLAG_COMPRESS_PAGE);

                blen = qemu_put_compression_data(rs->f, p, TARGET_PAGE_SIZE,

                                                 migrate_compress_level());

                if (blen > 0) {

                    rs->bytes_transferred += bytes_xmit + blen;

                    rs->norm_pages++;

                    pages = 1;

                } else {

                    qemu_file_set_error(rs->f, blen);

                    error_report("compressed data failed!");

                }

            }

            if (pages > 0) {

                ram_release_pages(block->idstr, offset, pages);

            }

        } else {

            pages = save_zero_page(rs, block, offset, p);

            if (pages == -1) {

                pages = compress_page_with_multi_thread(rs, block, offset);

            } else {

                ram_release_pages(block->idstr, offset, pages);

            }

        }

    }



    return pages;

}

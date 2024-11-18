static int ram_save_page(RAMState *rs, PageSearchStatus *pss, bool last_stage)

{

    int pages = -1;

    uint64_t bytes_xmit;

    ram_addr_t current_addr;

    uint8_t *p;

    int ret;

    bool send_async = true;

    RAMBlock *block = pss->block;

    ram_addr_t offset = pss->page << TARGET_PAGE_BITS;



    p = block->host + offset;

    trace_ram_save_page(block->idstr, (uint64_t)offset, p);



    /* In doubt sent page as normal */

    bytes_xmit = 0;

    ret = ram_control_save_page(rs->f, block->offset,

                           offset, TARGET_PAGE_SIZE, &bytes_xmit);

    if (bytes_xmit) {

        rs->bytes_transferred += bytes_xmit;

        pages = 1;

    }



    XBZRLE_cache_lock();



    current_addr = block->offset + offset;



    if (ret != RAM_SAVE_CONTROL_NOT_SUPP) {

        if (ret != RAM_SAVE_CONTROL_DELAYED) {

            if (bytes_xmit > 0) {

                rs->norm_pages++;

            } else if (bytes_xmit == 0) {

                rs->zero_pages++;

            }

        }

    } else {

        pages = save_zero_page(rs, block, offset, p);

        if (pages > 0) {

            /* Must let xbzrle know, otherwise a previous (now 0'd) cached

             * page would be stale

             */

            xbzrle_cache_zero_page(rs, current_addr);

            ram_release_pages(block->idstr, offset, pages);

        } else if (!rs->ram_bulk_stage &&

                   !migration_in_postcopy() && migrate_use_xbzrle()) {

            pages = save_xbzrle_page(rs, &p, current_addr, block,

                                     offset, last_stage);

            if (!last_stage) {

                /* Can't send this cached data async, since the cache page

                 * might get updated before it gets to the wire

                 */

                send_async = false;

            }

        }

    }



    /* XBZRLE overflow or normal page */

    if (pages == -1) {

        rs->bytes_transferred += save_page_header(rs, block,

                                                  offset | RAM_SAVE_FLAG_PAGE);

        if (send_async) {

            qemu_put_buffer_async(rs->f, p, TARGET_PAGE_SIZE,

                                  migrate_release_ram() &

                                  migration_in_postcopy());

        } else {

            qemu_put_buffer(rs->f, p, TARGET_PAGE_SIZE);

        }

        rs->bytes_transferred += TARGET_PAGE_SIZE;

        pages = 1;

        rs->norm_pages++;

    }



    XBZRLE_cache_unlock();



    return pages;

}

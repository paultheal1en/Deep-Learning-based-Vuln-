static bool get_next_page(GuestPhysBlock **blockptr, uint64_t *pfnptr,

                          uint8_t **bufptr, DumpState *s)

{

    GuestPhysBlock *block = *blockptr;

    hwaddr addr;

    uint8_t *buf;



    /* block == NULL means the start of the iteration */

    if (!block) {

        block = QTAILQ_FIRST(&s->guest_phys_blocks.head);

        *blockptr = block;

        assert(block->target_start % s->page_size == 0);

        assert(block->target_end % s->page_size == 0);

        *pfnptr = paddr_to_pfn(block->target_start);

        if (bufptr) {

            *bufptr = block->host_addr;

        }

        return true;

    }



    *pfnptr = *pfnptr + 1;

    addr = pfn_to_paddr(*pfnptr);



    if ((addr >= block->target_start) &&

        (addr + s->page_size <= block->target_end)) {

        buf = block->host_addr + (addr - block->target_start);

    } else {

        /* the next page is in the next block */

        block = QTAILQ_NEXT(block, next);

        *blockptr = block;

        if (!block) {

            return false;

        }

        assert(block->target_start % s->page_size == 0);

        assert(block->target_end % s->page_size == 0);

        *pfnptr = paddr_to_pfn(block->target_start);

        buf = block->host_addr;

    }



    if (bufptr) {

        *bufptr = buf;

    }



    return true;

}

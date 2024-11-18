static ram_addr_t get_start_block(DumpState *s)

{

    RAMBlock *block;



    if (!s->has_filter) {

        s->block = QTAILQ_FIRST(&ram_list.blocks);

        return 0;

    }



    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        if (block->offset >= s->begin + s->length ||

            block->offset + block->length <= s->begin) {

            /* This block is out of the range */

            continue;

        }



        s->block = block;

        if (s->begin > block->offset) {

            s->start = s->begin - block->offset;

        } else {

            s->start = 0;

        }

        return s->start;

    }



    return -1;

}

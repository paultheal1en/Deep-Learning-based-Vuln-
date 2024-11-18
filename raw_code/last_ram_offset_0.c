ram_addr_t last_ram_offset(void)

{

    RAMBlock *block;

    ram_addr_t last = 0;



    QTAILQ_FOREACH(block, &ram_list.blocks, next)

        last = MAX(last, block->offset + block->length);



    return last;

}

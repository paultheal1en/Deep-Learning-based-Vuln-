static ram_addr_t find_ram_offset(ram_addr_t size)

{

    RAMBlock *block, *next_block;

    ram_addr_t offset, mingap = ULONG_MAX;



    if (QLIST_EMPTY(&ram_list.blocks))

        return 0;



    QLIST_FOREACH(block, &ram_list.blocks, next) {

        ram_addr_t end, next = ULONG_MAX;



        end = block->offset + block->length;



        QLIST_FOREACH(next_block, &ram_list.blocks, next) {

            if (next_block->offset >= end) {

                next = MIN(next, next_block->offset);

            }

        }

        if (next - end >= size && next - end < mingap) {

            offset =  end;

            mingap = next - end;

        }

    }

    return offset;

}

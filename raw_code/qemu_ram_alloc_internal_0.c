RAMBlock *qemu_ram_alloc_internal(ram_addr_t size, ram_addr_t max_size,

                                  void (*resized)(const char*,

                                                  uint64_t length,

                                                  void *host),

                                  void *host, bool resizeable,

                                  MemoryRegion *mr, Error **errp)

{

    RAMBlock *new_block;

    Error *local_err = NULL;



    size = HOST_PAGE_ALIGN(size);

    max_size = HOST_PAGE_ALIGN(max_size);

    new_block = g_malloc0(sizeof(*new_block));

    new_block->mr = mr;

    new_block->resized = resized;

    new_block->used_length = size;

    new_block->max_length = max_size;

    assert(max_size >= size);

    new_block->fd = -1;

    new_block->host = host;

    if (host) {

        new_block->flags |= RAM_PREALLOC;

    }

    if (resizeable) {

        new_block->flags |= RAM_RESIZEABLE;

    }

    ram_block_add(new_block, &local_err);

    if (local_err) {

        g_free(new_block);

        error_propagate(errp, local_err);

        return NULL;

    }

    mr->ram_block = new_block;

    return new_block;

}

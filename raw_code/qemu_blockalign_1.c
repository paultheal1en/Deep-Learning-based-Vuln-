void *qemu_blockalign(BlockDriverState *bs, size_t size)

{

    return qemu_memalign((bs && bs->buffer_alignment) ? bs->buffer_alignment : 512, size);

}

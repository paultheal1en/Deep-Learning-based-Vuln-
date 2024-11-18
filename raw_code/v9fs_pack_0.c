size_t v9fs_pack(struct iovec *in_sg, int in_num, size_t offset,

                const void *src, size_t size)

{

    return v9fs_packunpack((void *)src, in_sg, in_num, offset, size, 1);

}

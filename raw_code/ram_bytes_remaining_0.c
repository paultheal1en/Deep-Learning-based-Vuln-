uint64_t ram_bytes_remaining(void)

{

    return ram_save_remaining() * TARGET_PAGE_SIZE;

}

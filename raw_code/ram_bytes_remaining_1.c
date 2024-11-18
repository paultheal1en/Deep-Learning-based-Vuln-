uint64_t ram_bytes_remaining(void)

{

    return ram_state->migration_dirty_pages * TARGET_PAGE_SIZE;

}

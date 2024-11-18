uint64_t blk_mig_bytes_remaining(void)

{

    return blk_mig_bytes_total() - blk_mig_bytes_transferred();

}

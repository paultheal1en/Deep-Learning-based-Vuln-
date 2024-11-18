int blk_mig_active(void)

{

    return !QSIMPLEQ_EMPTY(&block_mig_state.bmds_list);

}

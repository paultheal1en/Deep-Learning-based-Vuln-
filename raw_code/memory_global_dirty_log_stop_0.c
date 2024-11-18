void memory_global_dirty_log_stop(void)

{

    global_dirty_log = false;



    /* Refresh DIRTY_LOG_MIGRATION bit.  */

    memory_region_transaction_begin();

    memory_region_update_pending = true;

    memory_region_transaction_commit();



    MEMORY_LISTENER_CALL_GLOBAL(log_global_stop, Reverse);

}

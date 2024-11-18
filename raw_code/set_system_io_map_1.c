void set_system_io_map(MemoryRegion *mr)

{

    memory_region_transaction_begin();

    address_space_io.root = mr;

    memory_region_transaction_commit();

}

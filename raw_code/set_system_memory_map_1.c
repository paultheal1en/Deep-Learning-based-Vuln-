void set_system_memory_map(MemoryRegion *mr)

{

    memory_region_transaction_begin();

    address_space_memory.root = mr;

    memory_region_transaction_commit();

}

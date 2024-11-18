static void build_pci_bus_state_cleanup(AcpiBuildPciBusHotplugState *state)

{

    build_free_array(state->device_table);

    build_free_array(state->notify_table);

}

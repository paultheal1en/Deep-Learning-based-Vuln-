static void pc_machine_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);

    PCMachineClass *pcmc = PC_MACHINE_CLASS(oc);

    HotplugHandlerClass *hc = HOTPLUG_HANDLER_CLASS(oc);



    pcmc->inter_dimm_gap = true;

    pcmc->get_hotplug_handler = mc->get_hotplug_handler;

    mc->get_hotplug_handler = pc_get_hotpug_handler;

    mc->cpu_index_to_socket_id = pc_cpu_index_to_socket_id;

    mc->default_boot_order = "cad";

    mc->hot_add_cpu = pc_hot_add_cpu;

    mc->max_cpus = 255;

    mc->reset = pc_machine_reset;

    hc->plug = pc_machine_device_plug_cb;

    hc->unplug_request = pc_machine_device_unplug_request_cb;

    hc->unplug = pc_machine_device_unplug_cb;

}

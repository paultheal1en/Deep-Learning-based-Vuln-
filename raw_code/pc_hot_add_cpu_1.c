void pc_hot_add_cpu(const int64_t id, Error **errp)
{
    DeviceState *icc_bridge;
    int64_t apic_id = x86_cpu_apic_id_from_index(id);
    if (cpu_exists(apic_id)) {
        error_setg(errp, "Unable to add CPU: %" PRIi64
                   ", it already exists", id);
    if (id >= max_cpus) {
        error_setg(errp, "Unable to add CPU: %" PRIi64
                   ", max allowed: %d", id, max_cpus - 1);
    icc_bridge = DEVICE(object_resolve_path_type("icc-bridge",
                                                 TYPE_ICC_BRIDGE, NULL));
    pc_new_cpu(current_cpu_model, apic_id, icc_bridge, errp);
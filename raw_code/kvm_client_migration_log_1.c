static int kvm_client_migration_log(struct CPUPhysMemoryClient *client,

                                    int enable)

{

    return kvm_set_migration_log(enable);

}

void add_migration_state_change_notifier(Notifier *notify)

{

    notifier_list_add(&migration_state_notifiers, notify);

}

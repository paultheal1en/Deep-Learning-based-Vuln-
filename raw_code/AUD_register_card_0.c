void AUD_register_card (const char *name, QEMUSoundCard *card)

{

    audio_init ();

    card->name = qemu_strdup (name);

    memset (&card->entries, 0, sizeof (card->entries));

    LIST_INSERT_HEAD (&glob_audio_state.card_head, card, entries);

}

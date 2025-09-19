from notification_handler import NotificationHandler

def main():
    """
    Front-end interface for selecting languages and notifying traffic signs.
    """
    notification_handler = NotificationHandler()
    
    print("Welcome to the Traffic Sign Notification System!")
    print("Available Languages:")
    for i, lang in enumerate(notification_handler.available_languages, start=1):
        print(f"{i}. {Language.get(lang).display_name()} ({lang})")
    
    selected_languages = input("\nEnter the numbers of the languages you want to use (comma-separated): ")
    selected_languages = selected_languages.split(",")
    
    # Map selected numbers to language codes
    selected_language_codes = []
    for num in selected_languages:
        try:
            index = int(num.strip()) - 1
            if 0 <= index < len(notification_handler.available_languages):
                selected_language_codes.append(notification_handler.available_languages[index])
        except ValueError:
            continue
    
    if not selected_language_codes:
        print("No valid languages selected. Defaulting to all available languages.")
        selected_language_codes = notification_handler.available_languages
    
    detected_sign = input("\nEnter the detected traffic sign: ")
    notification_handler.notify_traffic_sign(detected_sign, selected_language_codes)

if __name__ == "__main__":
    main()

import random

subjects = [
    'Shah Rukh Khan',
    'Virat Kohli',
    'Nirmala Sitharaman',
    'Amitabh Bachchan',
    'Donald Trump',
    'PM Narendra Modi',
    'Akshay Kumar',
    'A LinkedIn User',
    'An Instagram Influencer',
    'A Facebook Uncle',
    'An AI Robot',
    'A Confused Coder'
]

actions = [
    'launches missiles at',
    'eats lunch with',
    'cancels collaboration with',
    'declares war on',
    'orders coffee for',
    'celebrates victory with',
    'dances awkwardly near',
    'accidentally unfollows',
    'challenges to a duel',
    'trains with'
]

places_or_things = [
    'at Red Fort',
    'in a Mumbai local train',
    'in space',
    'on Mars',
    'near Ganga Ghat',
    'during an IPL match',
    'inside Parliament',
    'over the sky',
    'in the metaverse',
    'inside a Zoom call'
]

headline_types = [
    "🌀 Conspiracy Alert",
    "🎉 Wholesome News",
    "🔥 Breaking Drama",
    "🤯 Absurd Report",
    "👽 Alien Invasion?",
    "📢 Trending Now"
]

intros = [
    "Earlier today,",
    "Just in:",
    "At midnight,",
    "Breaking from our sources —",
    "With no prior warning,",
    "In an unexpected twist,"
]

emojis = ['😂', '💥', '🚨', '🌶️', '👀', '🎭', '🎬', '👑', '🛸', '☕']

def generate_headline():
    subject = random.choice(subjects)
    action = random.choice(actions)
    place_or_thing = random.choice(places_or_things)
    headline_type = random.choice(headline_types)
    intro = random.choice(intros)
    emoji = random.choice(emojis)

    return f"{headline_type} {emoji}\n{intro} {subject} {action} {place_or_thing}."

def main():
    headlines = []
    print("📰 Welcome to the Fake News Headline Generator 2.0 📰")

    while True:
        count = input("\nHow many headlines do you want? (Enter a number or 'q' to quit): ").strip().lower()
        if count == 'q':
            break

        if not count.isdigit():
            print("Please enter a valid number.")
            continue

        for _ in range(int(count)):
            headline = generate_headline()
            print("\n" + headline)
            headlines.append(headline)

        save = input("\nDo you want to save these headlines to a file? (yes/no): ").strip().lower()
        if save == 'yes':
         with open("fake_news_headlines.txt", "a", encoding="utf-8") as file:
            for line in headlines:
             file.write(line + "\n")
             print("✅ Headlines saved to fake_news_headlines.txt")



        more = input("\nDo you want to generate more? (yes/no): ").strip().lower()
        if more != 'yes':
            break

    print("\nThanks for using the Fake News Headline Generator 2.0 ✨")

if __name__ == "__main__":
    main()

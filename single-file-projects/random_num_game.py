from random import randint

def random_num_game():
  print("This is a number guessing game. I pick a number and you have to guess.")
  print("But you can set an upper boundary!")
  boundary = int(input("Pick a number: "))
  if boundary <= 20:
    available_guesses = 5
  elif boundary <= 40:
    available_guesses = 8
  elif boundary <= 60:
    available_guesses = 10
  elif boundary <= 100:
    available_guesses = 15
  print(f"\nSo, I'll think of a number between 0 and {boundary}.")
  print(f"\nYou have {available_guesses} guesses.")
  random_num = randint(0, boundary)
  user_num = user_input()
  while user_num != random_num:
    if available_guesses > 1:
      if user_num > random_num:
        print("sorry, too high")
      elif user_num < random_num:
        print("sorry, too low")
      available_guesses -=1
      print(f"You have {available_guesses} guess(es) left.\n")
      user_num = user_input()
    else:
      print(f"Sorry, the correct number was {random_num}")
      break
  else:
    print(f"Hooray, you guessed the number {random_num} correctly!")


def user_input():
  return int(input("What's your guess? "))

random_num_game()
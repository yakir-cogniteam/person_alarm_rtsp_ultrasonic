pip install twilio


from twilio.rest import Client

account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token  = "your_auth_token"
client = Client(account_sid, auth_token)

message = client.messages.create(
    body="Person alarm!",
    from_="+1234567890",   # Your Twilio number
    to="+0987654321"       # Destination number
)

print(f"Message sent! SID: {message.sid}")
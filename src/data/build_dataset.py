"""Builds the labelled digital-arrest call corpus (English, Hindi, Marathi, Hinglish).

Methodology reference: no public benchmark dataset exists for Indian digital-arrest
interactions, so the corpus is constructed from documented scam-script patterns
(Indian cybercrime advisories / MHA statements), annotated with indicator families,
then augmented with realistic transcript noise.

Augmentation is deliberately applied per *base template* so that `group_id` can be
used for grouped splitting. This prevents near-duplicate variants of the same
underlying call from leaking across the train/test boundary.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from ..config import DATA_PROCESSED, LABEL_LEGIT, LABEL_SCAM, SEED

# (base_id, language, label, indicators, text)
SEEDS: list[tuple[str, str, int, tuple[str, ...], str]] = [
    # ---------------- SCAM: authority impersonation ----------------
    ("cbi_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "This is Senior Inspector Rao speaking from the Central Bureau of Investigation. There is a case registered against your name in Mumbai."),
    ("cbi_02", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat", "financial_coercion"),
     "I am calling from the Central Bureau of Investigation regarding money laundering charges filed against your account. You need to deposit the amount today."),
    ("ed_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "Madam, this is the Enforcement Directorate. A case number 45/2024 has been registered against you under section 420 IPC."),
    ("ed_02", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat", "urgency"),
     "I am from the Enforcement Directorate calling on behalf of the government. Your bank accounts have been frozen and you must respond immediately."),
    ("customs_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "Hello, I am a customs officer calling from the Department of Revenue Intelligence. We have intercepted a parcel in your name containing illegal substance."),
    ("customs_02", "en", LABEL_SCAM, ("authority_impersonation", "financial_coercion"),
     "This is the customs department. To release your parcel you must pay a customs duty of 85000 to this UPI ID right now."),
    ("cyber_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "Hello sir, I am calling from the cyber cell. Your Aadhaar card has been misused and there are 14 complaints registered against you."),
    ("incometax_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat", "financial_coercion"),
     "I am calling from the Income Tax Department. You have unpaid tax of 42000 rupees and if you do not pay today a warrant will be issued."),
    ("police_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat", "isolation_secrecy"),
     "This is the crime branch Delhi police. We have arrested your partner and we need your statement. Do not tell your family about this case."),
    ("rbi_01", "en", LABEL_SCAM, ("authority_impersonation", "financial_coercion"),
     "I am calling from the Reserve Bank of India. Your bank account has been involved in fraud. To keep the account active you must transfer the funds to this account."),
    ("secy_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat", "isolation_secrecy"),
     "Hello, I am the secretary of the Prime Minister speaking. This call is confidential, do not tell anyone. There is a serious case against you."),
    ("ncb_01", "en", LABEL_SCAM, ("authority_impersonation", "legal_threat", "urgency"),
     "I am calling from the National Crime Bureau. Your number came up in a drug trafficking investigation. Do not disconnect this call."),
    # ---------------- SCAM: legal threat ----------------
    ("legal_01", "en", LABEL_SCAM, ("legal_threat", "urgency"),
     "You have received a show cause notice under section 138 of the IT Act. You must respond within 24 hours or the case will be filed in court."),
    ("legal_02", "en", LABEL_SCAM, ("legal_threat", "isolation_secrecy", "financial_coercion"),
     "A criminal case has been filed against you for cheating. To avoid arrest you must deposit the money today. Keep this call between us."),
    ("legal_03", "en", LABEL_SCAM, ("legal_threat", "isolation_secrecy"),
     "Your son has been arrested. He is in serious trouble. Do not tell your wife. You cannot sleep until you pay the bail amount."),
    ("legal_04", "en", LABEL_SCAM, ("legal_threat", "financial_coercion", "urgency"),
     "You are under investigation for a fraud of 2 lakh rupees. Pay the penalty amount immediately to release your funds."),
    ("legal_05", "en", LABEL_SCAM, ("legal_threat", "isolation_secrecy"),
     "Sir I am a lawyer from the district court. Your property has a legal hold. Do not speak to anyone about this before the hearing."),
    # ---------------- SCAM: urgency ----------------
    ("urg_01", "en", LABEL_SCAM, ("urgency", "authority_impersonation"),
     "This is the CBI calling. Do not disconnect, the call is being recorded and your voice is being traced. You must act now."),
    ("urg_02", "en", LABEL_SCAM, ("urgency", "financial_coercion"),
     "Pay the amount right now. This is your last chance. Otherwise you will be arrested tonight."),
    ("urg_03", "en", LABEL_SCAM, ("urgency", "isolation_secrecy"),
     "Stay on the line and do not tell anyone. I need your cooperation immediately to clear your record."),
    # ---------------- SCAM: financial coercion ----------------
    ("fin_01", "en", LABEL_SCAM, ("financial_coercion", "urgency"),
     "Scan this QR code and pay 25000 to release your funds. Do it immediately, the offer closes in ten minutes."),
    ("fin_02", "en", LABEL_SCAM, ("financial_coercion", "authority_impersonation"),
     "I am from the cyber cell. Your money is trapped in a fake account. Send the amount to the UPI ID I am sharing now."),
    ("fin_03", "en", LABEL_SCAM, ("financial_coercion",),
     "Purchase a gift card of 50000 and share the code with me so we can clear your record."),
    ("fin_04", "en", LABEL_SCAM, ("financial_coercion", "legal_threat"),
     "The department has frozen your funds. Pay the security deposit of 75000 to your local station otherwise a warrant will be issued."),
    ("fin_05", "en", LABEL_SCAM, ("financial_coercion", "urgency", "isolation_secrecy"),
     "Transfer the money to this crypto wallet immediately. Do not tell your bank about this transaction."),
    # ---------------- SCAM: deepfake / family impersonation ----------------
    ("dfk_01", "en", LABEL_SCAM, ("isolation_secrecy", "financial_coercion", "urgency"),
     "Papa, papa I am in trouble, please send 50000 immediately I am in trouble do not tell mummy."),
    ("dfk_02", "en", LABEL_SCAM, ("isolation_seercion" if False else "isolation_secrecy", "financial_coercion"),
     "Aunty I am your son. I got into an accident and I need 80000 for the hospital right now, do not tell anyone at home."),
    # ---------------- SCAM: Hindi ----------------
    ("hi_cbi_01", "hi", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "मैं सीबीआई से बोल रहा हूँ, आपके खिलाफ मुंबई में मनी लॉन्ड्रिंग का मामला दर्ज किया गया है।"),
    ("hi_ed_01", "hi", LABEL_SCAM, ("authority_impersonation", "legal_threat", "urgency"),
     "मैं प्रवर्तन निदेशालय से बोल रहा हूँ, आपका बैंक खाता फ्रीज कर दिया गया है, आपको तुरंत जवाब देना होगा।"),
    ("hi_police_01", "hi", LABEL_SCAM, ("authority_impersonation", "legal_threat", "isolation_secrecy"),
     "मैं अपराध शाखा दिल्ली पुलिस से बोल रहा हूँ, आपके साथी को गिरफ्तार कर लिया गया है, अपनी पत्नी को मत बताइए।"),
    ("hi_cyber_01", "hi", LABEL_SCAM, ("authority_impersonation", "legal_threat", "financial_coercion"),
     "मैं साइबर सेल से बोल रहा हूँ, आपके आधार कार्ड का दुरुपयोग हुआ है, आपको 45000 जमा करने होंगे।"),
    ("hi_customs_01", "hi", LABEL_SCAM, ("authority_impersonation", "financial_coercion", "urgency"),
     "मैं सीमा शुल्क विभाग से बोल रहा हूँ, आपके पार्सल को जमा करने के लिए तुरंत 30000 भेजिए।"),
    ("hi_legal_01", "hi", LABEL_SCAM, ("legal_threat", "urgency"),
     "आपके विरुद्ध धोखाधड़ी का मामला दर्ज है, 24 घंटे में जवाब नहीं दिया तो अदालत में केस चलेगा।"),
    ("hi_urg_01", "hi", LABEL_SCAM, ("urgency", "authority_impersonation", "financial_coercion"),
     "फोन मत काटिए, कॉल रिकॉर्ड हो रही है, अभी यूपीआई आईडी पर पैसे भेजिए।"),
    ("hi_fin_01", "hi", LABEL_SCAM, ("financial_coercion", "urgency"),
     "इस क्यूआर कोड को स्कैन करके 25000 जमा कीजिए, केस यहीं पर बंद हो जाएगा।"),
    ("hi_iso_01", "hi", LABEL_SCAM, ("isolation_secrecy", "legal_threat", "financial_coercion"),
     "अपनी माता को बताइए मत, यह बात गोपनीय रखिए, ज़राना भर दीजिए।"),
    ("hi_fam_01", "hi", LABEL_SCAM, ("isolation_secrecy", "financial_coercion", "urgency"),
     "पापा, मुझे तुरंत 50000 भेज दो, मैं मुसीबत में हूँ, अम्मा को मत बताना।"),
    # ---------------- SCAM: Marathi ----------------
    ("mr_cbi_01", "mr", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "मी सीबीआय मधून बोलतो, तुमच्यावर पैसेाचा गुन्हा दाखल आहे, मुंबईत नोंद आहे."),
    ("mr_police_01", "mr", LABEL_SCAM, ("authority_impersonation", "legal_threat", "isolation_secrecy"),
     "मी गुन्हे शाखा पोलिस मधून बोलतो, तुमचा मित्राला अटक करण्यात आले आहे, कुटुंबाला सांगू नका."),
    ("mr_cyber_01", "mr", LABEL_SCAM, ("authority_impersonation", "legal_threat", "financial_coercion"),
     "मी सायबर सेल मधून बोलतो, तुमच्या खात्याचा दुरुपयोग झाला आहे, 45000 जमा करा."),
    ("mr_legal_01", "mr", LABEL_SCAM, ("legal_threat", "urgency"),
     "तुमच्यावर गुन्हा दाखल आहे, 24 तासांत उत्तर दिले नाही तर कोर्टात खोली होईल."),
    ("mr_urg_01", "mr", LABEL_SCAM, ("urgency", "authority_impersonation", "financial_coercion"),
     "फोन काटू नका, कॉल नोंदवली जात आहे, आत्ताच युपीआय वर पैसे पाठवा."),
    ("mr_fin_01", "mr", LABEL_SCAM, ("financial_coercion", "urgency"),
     "हा क्युआर कोड स्कॅन करा आणि 25000 जमा करा, तोच खरा होईल."),
    ("mr_iso_01", "mr", LABEL_SCAM, ("isolation_secrecy", "legal_threat"),
     "कोणाला सांगू नका, हे गोपनीय आहे, तुम्ही अडचणीत आहात."),
    ("mr_fam_01", "mr", LABEL_SCAM, ("isolation_secrecy", "financial_coercion", "urgency"),
     "बाबा, मला लगेच 50000 पाठवा, मी अडचणीत आहे, आईला सांगू नका."),
    # ---------------- SCAM: Hinglish / romanised ----------------
    ("hing_01", "hinglish", LABEL_SCAM, ("authority_impersonation", "legal_threat"),
     "Hello sir main CBI se bol raha hoon, aapke khilaf money laundering ka case darj hai."),
    ("hing_02", "hinglish", LABEL_SCAM, ("authority_impersonation", "financial_coercion", "urgency"),
     "Sir main cyber cell se bol raha hoon, abhi turant 40000 UPI id pe bhejo warna arrest ho jayega."),
    ("hing_03", "hinglish", LABEL_SCAM, ("isolation_secrecy", "urgency"),
     "Phone mat katiye, abhi kisi ko mat batayiye, main aapko secret bata raha hoon."),
    ("hing_04", "hinglish", LABEL_SCAM, ("financial_coercion", "urgency"),
     "Bhai jaldi se QR scan karke 20000 deposit kar do, offer khatam ho jayega."),
    ("hing_05", "hinglish", LABEL_SCAM, ("authority_impersonation", "legal_threat", "urgency"),
     "Hello main cyber crime cell se bol raha hoon, aapke naam se 3 lakh ka fraud hua hai, abhi case darj karna hai."),
    ("hing_06", "hinglish", LABEL_SCAM, ("isolation_secrecy", "financial_coercion"),
     "Aunty mujhe turant 60000 bhejo, main hospital me hu, kisi ko mat batao please."),
    ("hing_07", "hinglish", LABEL_SCAM, ("authority_impersonation", "legal_threat", "financial_coercion"),
     "Sir main ED se baat kar raha hoon, money laundering case hai, abhi 35000 UPI pe bhejo warna arrest."),
    ("hing_08", "hinglish", LABEL_SCAM, ("legal_threat", "urgency"),
     "Aapka account suspicious ho gaya hai, abhi OTP aur PAN share karo warna bank account band ho jayega."),
    # ---------------- LEGITIMATE ----------------
    ("legit_bank_otp_01", "en", LABEL_LEGIT, (),
     "Hello, this is HDFC Bank calling from your registered number. A transaction of 2500 rupees was declined at the merchant. Do you want to update your limit?"),
    ("legit_bank_otp_02", "en", LABEL_LEGIT, (),
     "Hi, ICICI Bank here. Your debit card ending 4421 is being used at a shop in Mumbai for 1200 rupees. Was this you?"),
    ("legit_bank_loan_01", "en", LABEL_LEGIT, (),
     "Good morning, this is State Bank credit card department. Your minimum amount due is 8400 and the due date is the 5th. Would you like a payment reminder on WhatsApp?"),
    ("legit_bank_fraud_01", "en", LABEL_LEGIT, (),
     "This is Kotak Mahindra fraud monitoring. We noticed a login from a new device in Pune. Please share the OTP we send to your registered mobile to block it."),
    ("legit_utility_01", "en", LABEL_LEGIT, (),
     "Hello, this is Maharashtra State Electricity Distribution. Your bill for the month of August is 2350 rupees and is due on the 20th. Pay through the app or UPI."),
    ("legit_water_01", "en", LABEL_LEGIT, (),
     "Dear customer, your water bill of 640 rupees is pending. You can pay at the municipal office or through the website."),
    ("legit_telecom_01", "en", LABEL_LEGIT, (),
     "Hi, this is Jio customer care. Your prepaid plan of 449 rupees expires in 5 days. Reply 1 to renew or 2 to talk to an agent."),
    ("legit_telecom_02", "en", LABEL_LEGIT, (),
     "Hello sir, your broadband installation is scheduled for tomorrow between 10 AM and 2 PM. Our engineer will call before arriving."),
    ("legit_delivery_01", "en", LABEL_LEGIT, (),
     "Your Amazon order 402-9981 is out for delivery and will arrive today between 6 PM and 8 PM. Track it in the app."),
    ("legit_doctor_01", "en", LABEL_LEGIT, (),
     "Hello, this is Dr Mehta clinic. Your appointment with Dr Mehta is confirmed on Thursday at 11 AM. Please bring your previous reports."),
    ("legit_gov_scheme_01", "en", LABEL_LEGIT, (),
     "Namaste, this is the PM Kisan helpline. Your eKYC is pending, please complete it on the portal or at your nearest Common Service Centre."),
    ("legit_gov_02", "en", LABEL_LEGIT, (),
     "This is the Nagpur Municipal Corporation. Property tax of 4800 rupees for the year 2026 is due. You may pay online or at the counter."),
    ("legit_insurance_01", "en", LABEL_LEGIT, (),
     "Good morning, this is LIC calling about your policy 4471882. Your premium of 3200 rupees is due on the 28th. You can pay via auto debit."),
    ("legit_gov_update_01", "en", LABEL_LEGIT, (),
     "This is an automated message from UIDAI. Your Aadhaar is not being updated as the address proof is missing. Please visit an Aadhaar Seva Kendra."),
    ("legit_support_01", "en", LABEL_LEGIT, (),
     "Thanks for contacting Flipkart support. Your return request has been approved and the refund of 899 rupees will be credited in 3 to 5 working days."),
    ("legit_loan_offer_01", "en", LABEL_LEGIT, (),
     "Hello, you have pre-approved offers from three lenders on our platform. You can compare the interest rate and apply online. No documents needed right now."),
    ("legit_police_112_01", "en", LABEL_LEGIT, (),
     "This is the 112 emergency response follow up. The complaint you filed yesterday at 8 PM has been assigned to patrol vehicle 42. Do you need an ambulance?"),
    ("legit_police_verify_01", "en", LABEL_LEGIT, (),
     "Hello, I am calling from Mumbai Traffic Police to remind you about the challan of 400 rupees pending on your vehicle. You can pay on the Parivahan website."),
    ("legit_bank_scam_awareness_01", "en", LABEL_LEGIT, (),
     "This is SBI calling to inform you about rising digital fraud. Banks never ask for your OTP or PIN on a call. Please stay alert and report to 1930 if you face any fraud."),
    # ---------------- LEGIT: Hindi ----------------
    ("hi_bank_01", "hi", LABEL_LEGIT, (),
     "नमस्ते, मैं एचडीएफसी बैंक से बोल रहा हूँ, आपका कार्ड अस्थायी रूप से ब्लॉक कर दिया गया है, कृपया अपना ओटीपी बताइए।"),
    ("hi_bank_02", "hi", LABEL_LEGIT, (),
     "आपका बैंक खाता जुड़ा है, आपका बैलेंस 5400 रुपये है, कोई अनुरोध हो तो बताइए।"),
    ("hi_utility_01", "hi", LABEL_LEGIT, (),
     "नमस्कार, यह बिजली विभाग का कॉल है, आपका बिल 1250 रुपये का है, 20 तारीख तक जमा कर दें।"),
    ("hi_gov_01", "hi", LABEL_LEGIT, (),
     "यह प्रधानमंत्री किसान हेल्पलाइन से कॉल है, आपका ईकेवाईसी अधूरा है, कृपया पोर्टल पर पूरा करें।"),
    ("hi_telecom_01", "hi", LABEL_LEGIT, (),
     "नमस्ते, यह रिलायंस जियो ग्राहक सेवा से है, आपका प्लान 5 दिन में समाप्त हो रहा है, नवीकरण करें।"),
    ("hi_doctor_01", "hi", LABEL_LEGIT, (),
     "नमस्ते, यह डॉ मेहता क्लिनिक से है, आपका अपॉइंटमेंट कल 11 बजे है, पुरानी रिपोर्ट साथ लाइए।"),
    ("hi_delivery_01", "hi", LABEL_LEGIT, (),
     "आपका अमेजन ऑर्डर डिलीवरी के लिए निकल चुका है, आज शाम 6 से 8 बजे के बीच आएगा।"),
    ("hi_support_01", "hi", LABEL_LEGIT, (),
     "फ्लिपकार्ट सहायता से धन्यवाद, आपकी रिटर्न रिक्वेस्ट स्वीकृत हो गई है, 899 रुपये 3 से 5 दिन में वापस आएंगे।"),
    # ---------------- LEGIT: Marathi ----------------
    ("mr_bank_01", "mr", LABEL_LEGIT, (),
     "नमस्कार, मी एचडीएफसी बँकतून बोलतो, तुमचे कार्ड तात्पुरता ब्लॉक केलेले आहे, कृपया ओटीपी सांगा."),
    ("mr_utility_01", "mr", LABEL_LEGIT, (),
     "नमस्कार, हे महाराष्ट्र राज्य विद्युत वितरण कंपनीचा कॉल आहे, तुमचे बिल 1250 रुपये आहे, 20 तारीखपर्यंत जमा करा."),
    ("mr_telecom_01", "mr", LABEL_LEGIT, (),
     "नमस्कार, हे जिओ ग्राहक सेवा आहे, तुमचा प्लॅन 5 दिवसांत संपतो, नूतनीकरण करा."),
    ("mr_gov_01", "mr", LABEL_LEGIT, (),
     "हे पीएम किसान हेल्पलाइनचा कॉल आहे, तुमचा ईकेवायसी अपूर्ण आहे, कृपया पोर्टलवर पूर्ण करा."),
    ("mr_support_01", "mr", LABEL_LEGIT, (),
     "फ्लिपकार्ट सहाय्यक कृपया यांचे आभार, तुमची रिटर््न विनंती मंजूर झाली आहे, 899 रुपये 3 ते 5 दिवसांत परत मिळतील."),
    ("mr_doctor_01", "mr", LABEL_LEGIT, (),
     "नमस्कार, हे डॉ मेहता क्लिनिक आहे, तुमची अपॉइंटमेंट उद्या सकाळी 11 वाजता आहे."),
    # ---------------- LEGIT: Hinglish ----------------
    ("hing_legit_01", "hinglish", LABEL_LEGIT, (),
     "Namaste sir, main HDFC bank se bol raha hoon, aapka EMI 4500 rupaya is mahine pending hai, aap pay kar sakte hai."),
    ("hing_legit_02", "hinglish", LABEL_LEGIT, (),
     "Hello main Jio se bol raha hoon, aapka data 2 GB khatam ho gaya hai, recharge kar lein."),
    ("hing_legit_03", "hinglish", LABEL_LEGIT, (),
     "Namaste main doctor se bol raha hoon, aapki appointment kal 5 baje hai, please time se aayein."),
    ("hing_legit_04", "hinglish", LABEL_LEGIT, (),
     "Hello main Axis Bank se bol raha hoon, aapka loan EMI 12000 rupaya pending hai, aap pay kar sakte hai."),
    ("hing_legit_05", "hinglish", LABEL_LEGIT, (),
     "Namaste main Airtel se bol raha hoon, aapka recharge 299 rupees 15 din me expire ho jayega."),
    ("hing_legit_06", "hinglish", LABEL_LEGIT, (),
     "Hello main Myntra se bol raha hoon, aapka order deliver ho gaya hai, 2 din me aayega."),
]

OPENERS = {
    "en": ["Hello, ", "Hi, ", "Good morning, ", "Good evening, ", "Hi there, "],
    "hi": ["नमस्ते, ", "नमस्कार, ", "हैलो, "],
    "mr": ["नमस्कार, ", "नमस्कार जी, ", "हॅलो, "],
    "hinglish": ["Hello, ", "Namaste, ", "Hi, ", "Hello sir, "],
}

CLOSERS = {
    "en": [" Thank you.", " Have a nice day.", " Please call back if you need help."],
    "hi": [" धन्यवाद।", " कृपया सहयोग करें।", " धन्यवाद।"],
    "mr": [" धन्यवाद.", " कृपया सहकार्य करा.", " धन्यवाद."],
    "hinglish": [" Thank you.", " Dhanyavad.", " Ok thanks."],
}

FILLERS = {
    "en": [" um, ", " so, ", " look, ", " basically, ", " you know, "],
    "hi": [" तो, ", " अरे, ", " सुनिए, "],
    "mr": [" तर, ", " अरे, ", " काळी, "],
    "hinglish": [" yaar, ", " suno, ", " basically, "],
}


def _augment(rng: random.Random, base: str, lang: str, group_id: str, variant: int) -> str:
    """Realistic transcript noise: fillers, openers, closers, truncation."""
    text = base
    if variant % 3 == 0:
        text = rng.choice(OPENERS[lang]) + text
    if variant % 4 == 1:
        pos = text.find(" ")
        cut = rng.randint(1, max(1, len(text) // 3))
        text = text[:cut] + rng.choice(FILLERS[lang]) + text[cut:].lstrip()
    if variant % 5 == 2:
        text = text + rng.choice(CLOSERS[lang])
    if variant % 7 == 3:
        text = text.lower()
    if variant % 11 == 4:
        text = text.replace(" ", "  ")
    return " ".join(text.split())


def build(n_variants: int = 4) -> Path:
    rng = random.Random(SEED)
    rows = []
    for base_id, lang, label, inds, text in SEEDS:
        rows.append(
            {
                "id": f"{base_id}_v0",
                "group_id": base_id,
                "variant": 0,
                "language": lang,
                "label": label,
                "indicators": "|".join(inds),
                "source": "advisory_pattern",
                "text": " ".join(text.split()),
            }
        )
        for v in range(1, n_variants):
            rows.append(
                {
                    "id": f"{base_id}_v{v}",
                    "group_id": base_id,
                    "variant": v,
                    "language": lang,
                    "label": label,
                    "indicators": "|".join(inds),
                    "source": "advisory_pattern+aug",
                    "text": _augment(rng, " ".join(text.split()), lang, base_id, v),
                }
            )

    out = DATA_PROCESSED / "calls.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scam = sum(1 for r in rows if r["label"] == LABEL_SCAM)
    groups = len({r["group_id"] for r in rows})
    print(f"wrote {out}  rows={len(rows)}  scam={scam}  legit={len(rows)-scam}  groups={groups}")
    for lg in sorted({r["language"] for r in rows}):
        sub = [r for r in rows if r["language"] == lg]
        s = sum(1 for r in sub if r["label"] == LABEL_SCAM)
        print(f"  {lg:9s} n={len(sub):4d}  scam={s:4d}  legit={len(sub)-s:4d}")
    return out


if __name__ == "__main__":
    build()
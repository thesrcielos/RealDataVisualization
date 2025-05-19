import unittest
import random
import numpy as np
from datetime import datetime
from algorithms import data_generator
from algorithms.data import symptoms_diagnoses, HOSPITALS_BOGOTA

class TestBogotaMedicalGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = data_generator.BogotaMedicalGenerator()
        random.seed(42)
        np.random.seed(42)

    def test_generate_age(self):
        ages = [self.generator.generate_age() for _ in range(100)]
        self.assertTrue(all(15 <= age <= 100 for age in ages))
        self.assertAlmostEqual(np.mean(ages), 35, delta=5)

    def test_generate_height(self):
        male_height = self.generator.generate_height(30, 'M')
        female_height = self.generator.generate_height(30, 'F')
        male_minor_height = self.generator.generate_height(15, 'M')
        elder_male_height = self.generator.generate_height(70, 'M')

        self.assertGreater(elder_male_height, male_minor_height)
        self.assertGreater(male_height, male_minor_height)
        self.assertGreater(male_height, female_height)
        self.assertTrue(150 <= male_height <= 200)
        self.assertTrue(145 <= female_height <= 200)
        self.assertIsNone(self.generator.generate_height(30, 'X'))

    def test_generate_weight(self):
        weight = self.generator.generate_weight(30, 'M', 175)
        self.assertTrue(50 <= weight <= 120)
        
        male_weight = self.generator.generate_weight(30, 'M', 170)
        female_weight = self.generator.generate_weight(30, 'F', 170)
        self.assertGreater(male_weight, female_weight)

        minor_weight = self.generator.generate_weight(15, 'M', 160)
        elder_weight = self.generator.generate_weight(70, 'M', 160)
        self.assertGreater(male_weight, minor_weight)
        self.assertGreater(male_weight, elder_weight)
        self.assertTrue(40 <= minor_weight <= 100)
        self.assertTrue(40 <= elder_weight <= 120)
    
    def test_get_bmi_category(self):
        bmi = self.generator.get_bmi_category(22.5)
        self.assertEqual(bmi, 'Normal')
        
        underweight_bmi = self.generator.get_bmi_category(17.5)
        overweight_bmi = self.generator.get_bmi_category(27.5)
        obese_bmi = self.generator.get_bmi_category(32.5)
        
        self.assertEqual(underweight_bmi, 'Underweight')
        self.assertEqual(overweight_bmi, 'Overweight')
        self.assertEqual(obese_bmi, 'Obese')

    def test_generate_blood_pressure(self):
        bp = self.generator.generate_blood_pressure(30, 22.5)
        systolic_part, diastolic_part = bp.split('/')
        systolic = int(systolic_part)
        diastolic = int(diastolic_part.split()[0])
        self.assertTrue(90 <= systolic <= 180)
        self.assertTrue(60 <= diastolic <= 120)

    def test_symptoms_diagnosis_generation(self):
        # Test different age groups
        for age in [15, 35, 65]:
            symptoms, diagnosis, chronic = self.generator.generate_symptoms_diagnosis(age, 18.5)
            self.assertIsInstance(symptoms, list)
            self.assertTrue(len(diagnosis[0]) >= 3)  
            
            # Verify diagnosis matches symptoms
            if 'Chequeo rutinario' not in symptoms:
                diagnoses = [d for symptom in symptoms for d in symptoms_diagnoses[symptom]['diagnoses']]
                self.assertIn(diagnosis, diagnoses)

    def test_full_patient_generation(self):
        patient = self.generator.generate_patient()
        
        # Test required fields
        required_fields = [
            'ID_Paciente', 'Nombre', 'Género', 'Edad', 'Peso (kg)',
            'Altura (cm)', 'IMC', 'Presión Arterial', 'Síntomas',
            'Diagnóstico (CIE-10)', 'Enfermedades Crónicas', 'Fecha Consulta',
            'Hospital', 'Dirección Hospital', 'Localidad',
            'Nivel Socioeconómico', 'Seguro Médico'
        ]
        for field in required_fields:
            self.assertIn(field, patient)

        # Test hospital data consistency
        hospital = patient['Hospital']
        self.assertIn(hospital, HOSPITALS_BOGOTA)
        self.assertEqual(patient['Dirección Hospital'], HOSPITALS_BOGOTA[hospital]['address'])
        self.assertEqual(patient['Localidad'], HOSPITALS_BOGOTA[hospital]['district'])

    def test_socioeconomic_distribution(self):
        levels = [self.generator.generate_socioeconomic_level() for _ in range(100)]
        low_count = levels.count('Bajo') / len(levels)
        self.assertAlmostEqual(low_count, 0.55, delta=0.1)
        medium_count = levels.count('Medio') / len(levels)
        self.assertAlmostEqual(medium_count, 0.35, delta=0.1)
        high_count = levels.count('Alto') / len(levels)
        self.assertAlmostEqual(high_count, 0.10, delta=0.1)


    def test_insurance_distribution(self):
        insurance = [self.generator.generate_health_insurance() for _ in range(100)]
        sisben_count = insurance.count('SISBÉN') / len(insurance)
        self.assertAlmostEqual(sisben_count, 0.42, delta=0.1)

        sura_count = insurance.count('EPS Sura') / len(insurance)
        self.assertAlmostEqual(sura_count, 0.15, delta=0.1)

        sanitas_count = insurance.count('Sanitas') / len(insurance)
        self.assertAlmostEqual(sanitas_count, 0.15, delta=0.1)

        nueva_eps_count = insurance.count('Nueva EPS') / len(insurance)
        self.assertAlmostEqual(nueva_eps_count, 0.18, delta=0.1)

    def test_date_generation(self):
        patient = self.generator.generate_patient()
        consult_date = datetime.strptime(patient['Fecha Consulta'], '%d/%m/%Y')
        self.assertLess(consult_date, datetime.now())
        self.assertGreater(consult_date, datetime(2021, 1, 1)) 


if __name__ == '__main__':
    unittest.main()
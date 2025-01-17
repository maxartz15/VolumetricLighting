﻿using UnityEngine;

public abstract class LightOverride : MonoBehaviour
{

	[Header("Overrides")]
	public float m_IntensityMult = 1.0f;
	[MinValue(0.0f)]
	public float m_RangeMult = 1.0f;

	public enum Type{None, Point, Tube, Area, Directional}

	Type m_Type = Type.None;
	bool m_Initialized = false;
	Light m_Light;

	public bool isOn
	{
		get
		{
			if (!isActiveAndEnabled)
				return false;

			Init();

			switch(m_Type)
			{
				case Type.Point: return m_Light.enabled || GetForceOn();
				case Type.Directional: return m_Light.enabled || GetForceOn();
			}

			return false;
		}

		private set{}
	}

	new public Light light {get{Init(); return m_Light;} private set{}}

	public Type type {get{Init(); return m_Type;} private set{}}

	// To get the "enabled" state check box
	void Update()
	{

	}

	public abstract bool GetForceOn();

	void Init()
	{
		if (m_Initialized)
			return;

		if ((m_Light = GetComponent<Light>()) != null)
		{
			switch(m_Light.type)
			{
				case LightType.Point: m_Type = Type.Point; break;
				case LightType.Directional: m_Type = Type.Directional; break;
				default: m_Type = Type.None; break;
			}
		}

		m_Initialized = true;
	}
}
